from typing import List, Tuple, Optional
import google.generativeai as genai
import numpy as np
from dotenv import load_dotenv
import os
import json
import re
import time
import ssl
import redis
from celery import Celery
from celery.schedules import crontab
from langchain_experimental.text_splitter import SemanticChunker
from langchain.text_splitter import RecursiveCharacterTextSplitter 
from langchain_huggingface import HuggingFaceEmbeddings 
from supabase import create_client, Client
from datetime import datetime, timedelta, timezone
from pymilvus import (
    connections,
    utility,
    FieldSchema,
    CollectionSchema,
    DataType,
    Collection,
)
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from dateutil import parser as date_parser

# Load environment variables from .env file
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY") 
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
ZILLIZ_URI = os.getenv("ZILLIZ_URI")
ZILLIZ_TOKEN = os.getenv("ZILLIZ_TOKEN")
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = os.getenv("REDIS_PORT", 6379)
REDIS_PASSWORD = os.getenv("REDIS_PASSWORD", "")
REDIS_SSL = os.getenv("REDIS_SSL", "true").lower() == "true"
USE_REDIS = os.getenv("USE_REDIS", "true").lower() == "true"

# Configure the Google Generative AI with the API key
genai.configure(api_key=GOOGLE_API_KEY)

# Redis Client Connection
if USE_REDIS:
    try:
        print(f"Connecting to Redis: host={REDIS_HOST}, port={REDIS_PORT}, ssl={REDIS_SSL}")
        ssl_context = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT) if REDIS_SSL else None
        if ssl_context:
            ssl_context.check_hostname = False
            ssl_context.verify_mode = ssl.CERT_NONE
        
        redis_client = redis.Redis(
            host=REDIS_HOST,
            port=int(REDIS_PORT),
            password=REDIS_PASSWORD if REDIS_PASSWORD else None,
            ssl=REDIS_SSL,
            ssl_cert_reqs=None,
            decode_responses=True,
            socket_timeout=5,
            socket_connect_timeout=10
        )
        redis_client.ping()
        print("Successfully connected to Redis.")
    except redis.RedisError as e:
        print(f"Failed to connect to Redis: {e}")
        redis_client = None
        USE_REDIS = False
else:
    redis_client = None
    print("Redis usage is disabled.")

# Celery App Configuration
broker_protocol = "rediss" if REDIS_SSL else "redis"
backend_protocol = "rediss" if REDIS_SSL else "redis"

celery_app = Celery(
    'chatbot',
    broker=f'{broker_protocol}://:{REDIS_PASSWORD or ""}@{REDIS_HOST}:{REDIS_PORT}/0',
    backend=f'{backend_protocol}://:{REDIS_PASSWORD or ""}@{REDIS_HOST}:{REDIS_PORT}/0',
    broker_use_ssl={'ssl_cert_reqs': ssl.CERT_NONE} if REDIS_SSL else None,
    redis_backend_use_ssl={'ssl_cert_reqs': ssl.CERT_NONE} if REDIS_SSL else None
)
celery_app.conf.update(
    timezone='Asia/Kolkata',
    enable_utc=False,
    beat_schedule={
        'nightly-update': {
            'task': 'chatbot.nightly_update_task',
            'schedule': crontab(hour=19, minute=49),
        }
    }
)

class InteractiveRAGChatbot:
    def __init__(self):
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        self.embedding_dim = 384  
        self.model = genai.GenerativeModel('gemini-1.5-flash')
        self.redis_client = redis_client if USE_REDIS else None
        self.in_memory_cache = {} if not USE_REDIS else None
        
        self.semantic_chunker = SemanticChunker(
            self.embeddings,
            breakpoint_threshold_type="percentile",
            breakpoint_threshold_amount=99 
        )
        
        self.recursive_chunker = RecursiveCharacterTextSplitter(
            chunk_size=700,
            chunk_overlap=50,
            separators=["\n\n", "\n", " ", ""]
        )

        self.supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
        
        self.contexts = {
            'products': {'collection': None},
            'updates': {'collection': None},
            'employees': {'collection': None},
            'departments': {'collection': None}
        }
        self.history = [] 
        
        try:
            connections.connect("default", uri=ZILLIZ_URI, token=ZILLIZ_TOKEN)
        except Exception as e:
            print(f"Failed to connect to Zilliz: {e}")
            raise 

    def _get_cache_key(self, prefix: str, key: str) -> str:
        return f"{prefix}:{key}"

    def _get_from_cache(self, key: str) -> Optional[any]:
        try:
            if self.redis_client:
                cached = self.redis_client.get(key)
                return json.loads(cached) if cached else None
            return self.in_memory_cache.get(key)
        except (redis.RedisError, TypeError, json.JSONDecodeError) as e:
            print(f"Error retrieving from cache with key '{key}': {e}")
            return None

    def _set_to_cache(self, key: str, value: any, ttl: int = 3600):
        try:
            if self.redis_client:
                self.redis_client.setex(key, ttl, json.dumps(value))
            elif self.in_memory_cache is not None:
                self.in_memory_cache[key] = value
        except (redis.RedisError, TypeError, json.JSONDecodeError) as e:
            print(f"Error setting cache for key '{key}': {e}")

    def _cache_embeddings(self, texts: List[str]) -> List[List[float]]:
        batch_hash = str(hash(tuple(texts)))
        cache_key = self._get_cache_key("embedding_batch", batch_hash)
        
        cached_embeddings = self._get_from_cache(cache_key)
        if cached_embeddings:
            # print("Returning cached embeddings.")
            return cached_embeddings
        
        embeddings = self.embeddings.embed_documents(texts)
        self._set_to_cache(cache_key, embeddings, ttl=86400)
        return embeddings

    def parse_query(self, query: str) -> Tuple[str, List[str]]:
        search_in_pattern = r"^(?:search\s*(?:for\s*)?.*?\s*in\s+([\w\s,]+?))(?:\s+and\s+([\w\s,]+))?\s*$"
        match = re.match(search_in_pattern, query.lower().strip())
        if match:
            contexts = []
            if match.group(1):
                contexts.extend([c.strip() for c in match.group(1).split(',')])
            if match.group(2):
                contexts.extend([c.strip() for c in match.group(2).split(',')])
            clean_query = re.sub(r"^(search\s*(?:for\s*)?.*?\s*in\s+.*$)", "", query, flags=re.IGNORECASE).strip()
            clean_query = clean_query or query
            return clean_query, [c for c in contexts if c in ['products', 'updates', 'employees', 'departments']]
        return query, []

    def determine_context(self, query: str) -> List[str]:
        product_keywords = ['product', 'eoxs', 'books', 'crm', 'people', 'shop', 'reports', 'features', 'software', 'pricing', 'trial', 'payment', 'inventory']
        update_keywords = ['team', 'project', 'status', 'blockers', 'tasks', 'deadline', 'progress', 'completed', 'working', 'updates', 'daily']
        employee_keywords = ['employee', 'department', 'manager', 'reporting', 'email', 'intern', 'ceo', 'staff', 'team member', 'role', 'job title', 'names', 'count', 'total', 'list', 'alphabetical', 'department id', 'departments']
        department_keywords = ['department', 'departments', 'department id', 'department name']
        scores = {
            'products': sum(1 for keyword in product_keywords if keyword.lower() in query.lower()),
            'updates': sum(1 for keyword in update_keywords if keyword.lower() in query.lower()) * 1.5, 
            'employees': sum(1 for keyword in employee_keywords if keyword.lower() in query.lower()) * 2.5,
            'departments': sum(1 for keyword in department_keywords if keyword.lower() in query.lower()) * 2.0
        }
        
        query_embedding = self.embeddings.embed_query(query)
        context_scores = {}
        for ctx in self.contexts:
            collection = self.contexts[ctx].get('collection')
            if collection and collection.has_index():
                search_params = {"metric_type": "L2", "params": {"nprobe": 10}}
                results = collection.search(data=[query_embedding], anns_field="embedding", param=search_params, limit=1, output_fields=[])
                if results and results[0]: 
                    distance = results[0].distances[0]
                    context_scores[ctx] = 1 / (1 + distance) 
                else:
                    context_scores[ctx] = 0 
        
        combined_scores = {ctx: scores[ctx] + context_scores.get(ctx, 0) for ctx in self.contexts}
        max_score = max(combined_scores.values()) if combined_scores else 0
        
        if max_score < 0.1: 
            if any(k in query.lower() for k in employee_keywords):
                return ['employees']
            elif any(k in query.lower() for k in update_keywords):
                return ['updates']
            else:
                return list(self.contexts.keys()) 
        
        return [ctx for ctx, score in combined_scores.items() if score > 0.1]

    def process_products_data(self, product: dict) -> str:
        text = f"Product: {product['title']}\nOverview: {product.get('overview', '')}\n"
        if product.get('main_feature_name'):
            text += f"Main Feature: {product['main_feature_name']}\nDetails: {product.get('main_feature_details', '')}\n"
        if product.get('industry_focus'):
            text += f"Industry Focus: {product['industry_focus']}\n"
        if product.get('category'):
            text += f"Category: {product['category']}\n"
        return text

    def process_daily_updates(self, updates: List[dict], employee_names: dict = None) -> str:
        texts = []
        employee_names = employee_names or {}
        for update in updates:
            emp_name = employee_names.get(update['employee_id'], 'Unknown')
            blockers_text = f"- {update['blockers'].strip()}" if update.get('blockers') and isinstance(update.get('blockers'), str) and update['blockers'].strip() else 'No blockers reported'
            tasks_text = f"- {update['update'].strip()}" if update.get('update') and isinstance(update.get('update'), str) and update['update'].strip() else 'No tasks reported'
            text = (
                f"Employee: {emp_name}\n"
                f"Employee ID: {update['employee_id']}\n"
                f"Date: {update['date']}\n"
                f"Team: {update['team'] or 'None'}\n"
                f"Project: {update['project'] or 'None'}\n"
                f"Tasks: {tasks_text}\n"
                f"Blockers: {blockers_text}\n"
            )
            texts.append(text)
        return "\n\n".join(texts)

    def process_employee_data(self, data: List[dict]) -> str:
        texts = []
        department_groups = {}
        for emp in data:
            department_info = emp.get('departments')
            dept = department_info['name'] if department_info and isinstance(department_info, dict) else 'Unknown'
            dept_id = emp.get('department_id', 'Unknown')
            if dept not in department_groups:
                department_groups[dept] = {'employees': [], 'department_id': dept_id}
            department_groups[dept]['employees'].append(emp)
        
        for department, info in department_groups.items():
            text = f"Department: {department}\nDepartment ID: {info['department_id']}\nEmployees:\n"
            for employee in info['employees']:
                text += (
                    f"Name: {employee['name']}\n"
                    f"Email: {employee['email']}\n"
                    f"Job Title: {employee['job_title']}\n"
                    f"Manager: {employee['manager'] or 'None'}\n"
                    f"Employee ID: {employee['employee_id']}\n"
                    f"Department ID: {employee['department_id']}\n"
                )
            texts.append(text)
        
        summary_text = "Department Summaries:\n"
        for department, info in department_groups.items():
            employee_count = len(info['employees'])
            managers = set(emp['manager'] for emp in info['employees'] if emp['manager'])
            summary_text += (
                f"Department: {department}\n"
                f"Department ID: {info['department_id']}\n"
                f"Total Employees: {employee_count}\n"
                f"Reporting Managers: {', '.join(managers) if managers else 'None'}\n"
            )
        texts.append(summary_text)
        return "\n\n".join(texts)
    
    def count_unique_employee_names(self) -> int:
        try:
            response = self.supabase.table('employees').select('name', count='exact').execute()
            return response.count
        except Exception as e:
            print(f"Error counting unique names: {str(e)}")
            return 0
            
    def get_employee_names(self, alphabetical: bool = False, with_details: bool = False) -> List:
        try:
            response = self.supabase.table('employees').select('name, email, job_title, manager, department_id, employee_id, departments!employees_department_id_fkey(name)').execute()
            employee_details = []
            for emp in response.data:
                department_info = emp.get('departments')
                department_name = department_info['name'] if department_info and isinstance(department_info, dict) else 'Unknown'
                detail = {
                    'name': emp['name'],
                    'details': (
                        f"Name: {emp['name']}\n"
                        f"Department: {department_name}\n"
                        f"Department ID: {emp.get('department_id', 'Unknown')}\n"
                        f"Job Title: {emp.get('job_title', 'Unknown')}\n"
                        f"Email: {emp.get('email', 'Unknown')}\n"
                        f"Manager: {emp.get('manager', 'None')}\n"
                        f"Employee ID: {emp.get('employee_id', 'Unknown')}\n"
                    )
                }
                employee_details.append(detail)
            if alphabetical:
                employee_details.sort(key=lambda x: x['name'].lower())
            return [d['details'] for d in employee_details] if with_details else [d['name'] for d in employee_details]
        except Exception as e:
            print(f"Error retrieving employee names: {str(e)}")
            return []

    def get_department_id(self, dept_name: str) -> str:
        try:
            response = self.supabase.table('departments').select('department_id').ilike('name', dept_name).execute()
            if response.data:
                return f"[EMPLOYEES Context] The department ID for {dept_name} is {response.data[0]['department_id']}."
            return f"[EMPLOYEES Context] No department found with the name '{dept_name}'."
        except Exception as e:
            return f"[EMPLOYEES Context] Error retrieving department ID for '{dept_name}': {str(e)}."

    def list_departments(self) -> str:
        try:
            response = self.supabase.table('departments').select('name, department_id').execute()
            if response.data:
                dept_list = "\n".join(f"- {dept['name']}: {dept['department_id']}" for dept in response.data)
                return f"[EMPLOYEES Context] List of departments with their IDs:\n{dept_list}"
            return "[EMPLOYEES Context] No departments found."
        except Exception as e:
            return f"[EMPLOYEES Context] Error listing departments: {str(e)}."

    def get_updates_for_employee(self, emp_name: str, days_ago: int = None) -> str:
        try:
            emp_response = self.supabase.table('employees').select('employee_id, name').ilike('name', f'%{emp_name}%').execute()
            if not emp_response.data:
                return f"[UPDATES Context] No employee found with the name '{emp_name}'."
            
            employee_ids = [emp['employee_id'] for emp in emp_response.data]
            
            updates_query = self.supabase.table('daily_updates').select('employee_id, date, team, project, update, blockers').in_('employee_id', employee_ids)
            if days_ago is not None:
                cutoff_date = (datetime.now() - timedelta(days=days_ago)).date()
                updates_query = updates_query.gte('date', cutoff_date)
            
            updates_response = updates_query.execute()
            if not updates_response.data:
                time_frame = f" in the last {days_ago} days" if days_ago else ""
                return f"[UPDATES Context] No updates found for employee '{emp_name}'{time_frame}."
            
            employee_names_map = {emp['employee_id']: emp['name'] for emp in emp_response.data}
            
            processed_updates = self.process_daily_updates(updates_response.data, employee_names_map)
            
            time_frame = f" in the last {days_ago} days" if days_ago else ""
            return f"[UPDATES Context] Updates for '{emp_name}'{time_frame}:\n" + processed_updates
        except Exception as e:
            return f"[UPDATES Context] Error retrieving updates for '{emp_name}': {str(e)}."

    def get_completed_tasks_by_department(self, dept_name: str) -> str:
        try:
            dept_response = self.supabase.table('departments').select('department_id').ilike('name', dept_name).execute()
            if not dept_response.data:
                return f"[UPDATES + EMPLOYEES Context] No department found with the name '{dept_name}'."
            dept_id = dept_response.data[0]['department_id']

            emp_response = self.supabase.table('employees').select('employee_id, name, email, job_title, manager').eq('department_id', dept_id).execute()
            if not emp_response.data:
                return f"[UPDATES + EMPLOYEES Context] No employees found in department '{dept_name}'."
            
            employee_ids = [emp['employee_id'] for emp in emp_response.data]
            employee_details = {emp['employee_id']: emp for emp in emp_response.data}

            updates_response = self.supabase.table('daily_updates').select('employee_id, date, project, update').in_('employee_id', employee_ids).execute()
            if not updates_response.data:
                return f"[UPDATES + EMPLOYEES Context] No updates found for employees in department '{dept_name}'."

            emp_tasks = {}
            for update in updates_response.data: 
                emp_id = update['employee_id']
                if emp_id not in emp_tasks:
                    emp_tasks[emp_id] = []
                tasks_list = update['update'] if isinstance(update['update'], list) else []
                for task in tasks_list:
                    if 'completed' in task.lower():
                        emp_tasks[emp_id].append((task, update['project'] or 'Unknown', update['date']))

            texts = []
            for emp_id in sorted(emp_tasks.keys(), key=lambda x: employee_details.get(x, {}).get('name', '').lower()):
                emp = employee_details.get(emp_id, {})
                name = emp.get('name', f"Unknown (Employee ID: {emp_id})")
                tasks = emp_tasks.get(emp_id, [])
                if not tasks:
                    continue 
                text = (
                    f"- **{name} ({emp_id})**:\n"
                    f"  - Completed Tasks:\n"
                    f"    - " + "\n    - ".join(f"{task} ({project}, {date})" for task, project, date in tasks) + "\n"
                    f"  - Details:\n"
                    f"    - Email: {emp.get('email', 'Not available')}\n"
                    f"    - Job Title: {emp.get('job_title', 'Not available')}\n"
                    f"    - Manager: {emp.get('manager', 'None')}\n"
                )
                texts.append(text)

            if not texts:
                return f"[UPDATES + EMPLOYEES Context] No completed tasks found for employees in department '{dept_name}'."
            
            return f"[UPDATES + EMPLOYEES Context] Completed tasks and details for employees in '{dept_name}':\n" + "\n".join(texts)
        except Exception as e:
            return f"[UPDATES + EMPLOYEES Context] Error retrieving completed tasks for '{dept_name}': {str(e)}."

    def validate_data_integrity(self):
        try:
            updates_response = self.supabase.table('daily_updates').select('employee_id').execute()
            update_emp_ids = set(update['employee_id'] for update in updates_response.data)
            emp_response = self.supabase.table('employees').select('employee_id').execute()
            emp_ids = set(emp['employee_id'] for emp in emp_response.data)
            orphaned_ids = update_emp_ids - emp_ids
            if orphaned_ids:
                print(f"Warning: Found {len(orphaned_ids)} orphaned employee IDs in daily_updates: {orphaned_ids}")
        except Exception as e:
            print(f"Error validating data integrity: {str(e)}")

    def _get_or_create_collection(self, collection_name: str) -> Collection:
        if utility.has_collection(collection_name):
            return Collection(collection_name)

        fields = [
            FieldSchema(name="pk", dtype=DataType.VARCHAR, is_primary=True, auto_id=True, max_length=100),
            FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535), 
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=self.embedding_dim)
        ]
        if collection_name == "employees":
            fields.extend([
                FieldSchema(name="employee_id", dtype=DataType.VARCHAR, max_length=50),
                FieldSchema(name="name", dtype=DataType.VARCHAR, max_length=100),
                FieldSchema(name="team", dtype=DataType.VARCHAR, max_length=50),
                FieldSchema(name="role", dtype=DataType.VARCHAR, max_length=100),
            ])

        schema = CollectionSchema(fields, f"Collection for {collection_name}")
        collection = Collection(collection_name, schema)

        index_params = {"metric_type": "L2", "index_type": "HNSW", "params": {"M": 8, "efConstruction": 200}}
        collection.create_index("embedding", index_params)
        print(f"Created new Milvus collection: {collection_name}")
        return collection

    def load_data_from_supabase(self, table_name: str, select_columns: str, page_size: int = 1000):
        cache_key = self._get_cache_key("supabase_query", f"{table_name}:{select_columns}")
        cached_data = self._get_from_cache(cache_key)
        if cached_data:
            # print(f"Returning cached Supabase data for table: {table_name}")
            return cached_data

        all_records = []
        offset = 0
        try:
            while True:
                response = self.supabase.table(table_name).select(select_columns).range(offset, offset + page_size - 1).execute()
                if not response.data:
                    break
                all_records.extend(response.data)
                offset += page_size
                if len(response.data) < page_size:
                    break
                time.sleep(0.1)
            
            ttl = 7200 if 'updates' in table_name else 86400
            self._set_to_cache(cache_key, all_records, ttl=ttl)
            return all_records
        except Exception as e:
            print(f"Error fetching data from Supabase for table '{table_name}': {e}")
            raise

    def _safe_join_content(self, content_list, default_message: str = "No content available") -> str:
        if not content_list:
            return default_message
        try:
            cleaned_items = []
            for item in content_list:
                if item is None:
                    continue
                elif isinstance(item, str):
                    cleaned_item = item.strip()
                    if cleaned_item:
                        cleaned_items.append(cleaned_item)
                elif isinstance(item, dict):
                    dict_str = ', '.join(f"{k}: {v}" for k, v in item.items() if v is not None)
                    if dict_str:
                        cleaned_items.append(dict_str)
                else:
                    str_item = str(item).strip()
                    if str_item:
                        cleaned_items.append(str_item)
            if not cleaned_items:
                return default_message
            result = ' | '.join(cleaned_items)
            result = re.sub(r'\s+', ' ', result).strip()
            return result if result else default_message
        except Exception as e:
            return default_message
    
    def nightly_update_of_daily_updates(self):
        print("Starting scheduled refresh of daily updates...")
        if utility.has_collection('updates'):
            utility.drop_collection('updates')
            print("Dropped existing 'updates' collection for refresh.")
        
        updates_collection = self._get_or_create_collection('updates')
        
        emp_response = self.load_data_from_supabase('employees', 'employee_id, name')
        employee_names = {emp['employee_id']: emp['name'] for emp in emp_response}
        
        since_timestamp = (datetime.now(timezone.utc) - timedelta(days=1)).isoformat()
        
        since_cache_key = self._get_cache_key("supabase_query", f"daily_updates:employee_id, date, team, project, update, blockers, created_at:since:{since_timestamp[:10]}")
        if self.redis_client: self.redis_client.delete(since_cache_key)

        all_updates_data = self.load_data_from_supabase('daily_updates', 'employee_id, date, team, project, update, blockers, created_at')
        
        recent_updates = [
            update for update in all_updates_data 
            if update.get('created_at') and datetime.fromisoformat(update['created_at'].replace('Z', '+00:00')) > datetime.fromisoformat(since_timestamp)
        ]
        
        print(f"Fetched {len(recent_updates)} new daily update records from the last 24 hours.")

        if not recent_updates:
            print("No new updates to load.")
            self.contexts['updates']['collection'] = updates_collection
            updates_collection.load()
            return
            
        all_chunks_from_updates = []
        for update_record in tqdm(recent_updates, desc="Processing new daily updates"):
            record_text = self.process_daily_updates([update_record], employee_names)
            chunks_from_record = self.recursive_chunker.split_text(record_text)
            all_chunks_from_updates.extend(chunks_from_record)
        
        if all_chunks_from_updates:
            embeddings = self._cache_embeddings(all_chunks_from_updates)
            updates_collection.insert([all_chunks_from_updates, embeddings])
            updates_collection.flush()
            print(f"Loaded {len(all_chunks_from_updates)} new chunks into the 'updates' collection.")
        
        self.contexts['updates']['collection'] = updates_collection
        updates_collection.load()
        print("Scheduled refresh of daily updates completed.")

    def load_all_data(self):
        load_start_time = time.perf_counter()
        try:
            self.validate_data_integrity()

            # Products Data
            products_collection = self._get_or_create_collection('products')
            if products_collection.num_entities == 0:
                print("Loading products data...")
                all_products_data = self.load_data_from_supabase('products', '*')
                print(f"Fetched {len(all_products_data)} product records.")
                product_chunks = [self.process_products_data(product) for product in all_products_data]
                embeddings = self._cache_embeddings(product_chunks)
                products_collection.insert([product_chunks, embeddings])
                products_collection.flush()
                print(f"Processed and inserted {len(product_chunks)} chunks for products.")
            self.contexts['products']['collection'] = products_collection
            products_collection.load()
            print(f"Products collection loaded with {products_collection.num_entities} entities.")

            # Employees Data
            employees_collection = self._get_or_create_collection('employees')
            if employees_collection.num_entities == 0:
                print("Loading employees data...")
                all_employees_data = self.load_data_from_supabase(
                    'employees',
                    'name, email, job_title, manager, employee_id, department_id, departments!employees_department_id_fkey(name)'
                )
                print(f"Fetched {len(all_employees_data)} employee records.")
                employee_chunks = []
                metadata_fields = {"employee_ids": [], "names": [], "teams": [], "roles": []}
                for emp in all_employees_data:
                    name = emp.get("name", "Unknown")
                    department_info = emp.get("departments")
                    department_name = department_info.get("name", "Unknown") if department_info and isinstance(department_info, dict) else "Unknown"
                    job_title = emp.get("job_title", "Unknown")
                    emp_id = emp.get("employee_id", "Unknown")
                    text = (
                        f"Name: {name}\n"
                        f"Email: {emp.get('email', 'Unknown')}\n"
                        f"Job Title: {job_title}\n"
                        f"Manager: {emp.get('manager', 'Unknown')}\n"
                        f"Employee ID: {emp_id}\n"
                        f"Department ID: {emp.get('department_id', 'Unknown')}\n"
                        f"Department Name: {department_name}\n"
                    )
                    employee_chunks.append(text)
                    metadata_fields["employee_ids"].append(emp_id)
                    metadata_fields["names"].append(name)
                    metadata_fields["teams"].append(department_name)
                    metadata_fields["roles"].append(job_title)
                embeddings = self._cache_embeddings(employee_chunks)
                employees_collection.insert([employee_chunks, embeddings, metadata_fields["employee_ids"], metadata_fields["names"], metadata_fields["teams"], metadata_fields["roles"]])
                employees_collection.flush()
                print(f"Processed and inserted {len(employee_chunks)} chunks for employees.")
            self.contexts['employees']['collection'] = employees_collection
            employees_collection.load()
            print(f"Employees collection loaded with {employees_collection.num_entities} entities.")
            
            # Departments Data
            departments_collection = self._get_or_create_collection('departments')
            if departments_collection.num_entities == 0:
                print("Loading departments data...")
                all_departments_data = self.load_data_from_supabase('departments', '*')
                print(f"Fetched {len(all_departments_data)} department records.")
                department_chunks = [f"Department: {dept.get('name', 'Unknown')}\nDepartment ID: {dept.get('department_id', 'Unknown')}\n" for dept in all_departments_data]
                embeddings = self._cache_embeddings(department_chunks)
                departments_collection.insert([department_chunks, embeddings])
                departments_collection.flush()
                print(f"Processed and inserted {len(department_chunks)} chunks for departments.")
            self.contexts['departments']['collection'] = departments_collection
            departments_collection.load()
            print(f"Departments collection loaded with {departments_collection.num_entities} entities.")

            # Updates Data (Initial Load)
            updates_collection = self._get_or_create_collection('updates')
            if updates_collection.num_entities == 0:
                print("\nPerforming initial load of all daily updates...")
                emp_response = self.load_data_from_supabase('employees', 'employee_id, name')
                employee_names = {emp['employee_id']: emp['name'] for emp in emp_response}
                all_updates_data = self.load_data_from_supabase('daily_updates', 'employee_id, date, team, project, update, blockers, created_at')
                print(f"Fetched {len(all_updates_data)} total daily update records.")
                
                all_chunks_from_updates = []
                for update_record in tqdm(all_updates_data, desc="Processing all daily updates"):
                    record_text = self.process_daily_updates([update_record], employee_names)
                    chunks_from_record = self.recursive_chunker.split_text(record_text)
                    all_chunks_from_updates.extend(chunks_from_record)
                
                # --- FIX: Process large data in batches to prevent timeouts ---
                if all_chunks_from_updates:
                    print(f"Embedding and inserting {len(all_chunks_from_updates)} chunks in batches...")
                    batch_size = 1024
                    total_chunks_inserted = 0
                    for i in tqdm(range(0, len(all_chunks_from_updates), batch_size), desc="Embedding/Inserting Batches"):
                        batch_chunks = all_chunks_from_updates[i:i + batch_size]
                        batch_embeddings = self._cache_embeddings(batch_chunks)
                        if batch_embeddings:
                            try:
                                updates_collection.insert([batch_chunks, batch_embeddings])
                                total_chunks_inserted += len(batch_chunks)
                            except Exception as e:
                                print(f"Error inserting batch into Milvus: {e}")
                    updates_collection.flush()
                    print(f"Completed initial load of {total_chunks_inserted} chunks for daily updates.")
            self.contexts['updates']['collection'] = updates_collection
            updates_collection.load()
            print(f"Updates collection loaded with {updates_collection.num_entities} entities.")

        except Exception as e:
            print(f"Error loading data into Zilliz: {e}")
            raise
        load_end_time = time.perf_counter()
        print(f"Overall data loading completed in {load_end_time - load_start_time:.2f} seconds.")

    def get_relevant_chunks(self, query: str, contexts: List[str], k: int = 200) -> Tuple[List[str], List[str]]:
        query_embedding = self.embeddings.embed_query(query)
        all_chunks = []
        for ctx in contexts or self.contexts.keys():
            collection = self.contexts[ctx].get('collection')
            if collection and collection.has_index():
                search_params = {"metric_type": "L2", "params": {"ef": 128}} 
                results = collection.search(data=[query_embedding], anns_field="embedding", param=search_params, limit=k, output_fields=["text"])
                if results and results[0]:
                    for hit in results[0]: 
                        if hit.distance < 1.5: 
                            all_chunks.append({'text': hit.entity.get('text'), 'distance': hit.distance, 'context': ctx})
        all_chunks.sort(key=lambda x: x['distance'])
        top_chunks = [chunk['text'] for chunk in all_chunks[:k]]
        used_contexts = list(set(chunk['context'] for chunk in all_chunks[:k]))
        return top_chunks, used_contexts

    def create_chatgpt_style_prompt(self, context: str, query: str, intent: str, history: str = "") -> str:
        role_map = {
            "comparison": "You are a product analyst, skilled at comparing features and explaining differences in a clear, friendly way.",
            "definition": "You are a teacher, great at explaining concepts simply and clearly.",
            "person_info": "You are an HR assistant, cheerful and helpful, ready to provide information about people and teams.",
            "generic": "You are KYNEX, a helpful, cheerful, and knowledgeable assistant."
        }
        role = role_map.get(intent, role_map["generic"])
        assistant_name = "KYNEX"
        prompt = f"""
# {assistant_name} Chatbot

{role}

*Instructions:*
- Respond in a friendly, conversational, and complete way, just like ChatGPT would.
- Use Markdown formatting: headers, bold, bullet points, etc.
- Add a cheerful tone and sign off as '{assistant_name}'.
- If you don't have enough information, politely say so.
"""
        if intent == "comparison":
            prompt += "\n- *For this question, always present the answer as a Markdown table comparing the main features, tasks, and focus areas. If a table is not possible, use bullet points.*\n"
        elif intent == "person_info":
            prompt += "\n- *For this question, prefer a Markdown bullet list for details about people, teams, or roles.*\n"
        elif intent == "definition":
            prompt += "\n- *For this question, provide a concise paragraph and, if helpful, a short bullet list for key points.*\n"
        else:
            prompt += "\n- *For this question, use the most suitable format (paragraph, bullets, or table) for clarity.*\n"
        
        if history:
            prompt += f"\n---\n*Recent Conversation:*\n{history}\n---\n"
        
        prompt += f"\n*Context:*\n{context}\n\n*User Question:* {query}\n\n*Your Answer:*"
        return prompt

    def detect_intent(self, query: str) -> str:
        query_lower = query.lower()
        if any(word in query_lower for word in ["compare", "vs", "difference", "contrast"]):
            return "comparison"
        elif any(word in query_lower for word in ["what is", "define", "explain", "definition"]):
            return "definition"
        elif any(word in query_lower for word in ["employee", "team", "member", "person", "staff", "names", "details", "department id", "departments", "updates on", "tasks completed"]):
            return "person_info"
        return "generic"

    def extract_name_and_date(self, query):
        date_match = re.search(r"on ([^?]+)", query.lower())
        date = None
        if date_match:
            date_str = date_match.group(1).strip(" .?")
            try:
                parsed_date = date_parser.parse(date_str, fuzzy=True)
                date = parsed_date.strftime("%Y-%m-%d")
            except Exception:
                date = None
        name_match = re.search(r"what did ([a-zA-Z .'-]+) do on", query.lower())
        emp_name = None
        if name_match:
            emp_name = name_match.group(1).strip()
        return emp_name, date
    
    def get_yearly_updates_raw_for_employee(self, emp_name: str, year: int) -> str:
        try:
            emp_response = self.supabase.table('employees').select('employee_id, name').ilike('name', f'%{emp_name}%').execute()
            if not emp_response.data:
                return f"[YEARLY UPDATES Context] No employee found with the name '{emp_name}'."

            employee_ids = [emp['employee_id'] for emp in emp_response.data]

            start_date = f"{year}-01-01"
            end_date = f"{year}-12-31"

            updates_query = self.supabase.table('daily_updates').select('employee_id, date, team, project, update, blockers').in_('employee_id', employee_ids).gte('date', start_date).lte('date', end_date)

            updates_response = updates_query.execute()
            if not updates_response.data:
                return f"[YEARLY UPDATES Context] No updates found for '{emp_name}' in {year}."

            texts = []
            for update in sorted(updates_response.data, key=lambda x: x['date']):
                tasks_text = update['update'] or 'No tasks reported'
                blockers_text = update['blockers'] or 'No blockers reported'
                text = (
                    f"Employee: {emp_name}\n"
                    f"Date: {update['date']}\n"
                    f"Team: {update['team'] or 'None'}\n"
                    f"Project: {update['project'] or 'None'}\n"
                    f"Tasks: {tasks_text}\n"
                    f"Blockers: {blockers_text}\n"
                )
                texts.append(text)
            return "\n".join(texts)
        except Exception as e:
            return f"[YEARLY UPDATES Context] Error retrieving raw updates for '{emp_name}': {str(e)}."

    def answer_query(self, query: str) -> str:
        if not query.strip():
            return "Please provide a valid query."
        if not any(self.contexts[ctx]['collection'] for ctx in self.contexts):
            return "No data loaded into Zilliz. Please check the connection and run load_all_data()."
        
        cache_key = self._get_cache_key("final_answer", query)
        cached_response = self._get_from_cache(cache_key)
        if cached_response:
            print("Returning cached answer for query.")
            return cached_response

        try:
            clean_query, explicit_contexts = self.parse_query(query)
            query_to_use = clean_query or query
            self.history.append({'query': query_to_use, 'context_types': [], 'response': None})
            if len(self.history) > 100:
                self.history = self.history[-100:]
            query_lower = query_to_use.lower()

            yearly_pattern = r"(?:yearly|annual|monthly|month) updates? for ([a-zA-Z .'-]+) (?:for|in) (\d{4})"
            match = re.search(yearly_pattern, query_lower)
            if match:
                emp_name = match.group(1).strip()
                year = int(match.group(2))
                raw_context = self.get_yearly_updates_raw_for_employee(emp_name, year)
                user_prompt = (
                    f"You are an assistant. Given the following daily updates for {emp_name} in {year}, "
                    f"generate a clear monthly summary. For EACH MONTH, write exactly 2 bullet points "
                    f"highlighting the most important tasks, themes, or blockers. "
                    f"Follow a chronological order, be as impartial as possible, "
                    f"and ensure all 12 months are mentioned, using 'No significant updates' if no data exists.\n\n"
                    f"{raw_context}"
                )
                response = self.model.generate_content(user_prompt)
                answer = response.text
                self.history[-1]['context_types'] = ['updates']
                self.history[-1]['response'] = answer
                self._set_to_cache(cache_key, answer, ttl=7200)
                return answer

            tasks_dept_pattern = r"employee name wise from\s+(.+?)\s+give tasks completed.*details"
            match = re.search(tasks_dept_pattern, query_lower)
            if match:
                dept_name = match.group(1).strip()
                answer = self.get_completed_tasks_by_department(dept_name)
                self.history[-1]['context_types'] = ['updates', 'employees'] 
                self.history[-1]['response'] = answer
                self._set_to_cache(cache_key, answer, ttl=7200)
                return answer

            updates_pattern = r"(?:give me\s*|get\s*|show\s*)updates?\s*(?:of\s*(?:last\s*)?(\d+\s*(?:week|day)s?)\s*)?(?:on|for)\s+(.+)"
            match = re.search(updates_pattern, query_lower)
            if match:
                time_frame = match.group(1)
                emp_name = match.group(2).strip()
                days_ago = None
                if time_frame:
                    num = int(re.search(r'\d+', time_frame).group())
                    unit = time_frame.lower().replace(str(num), '').strip()
                    days_ago = num * 7 if 'week' in unit else num 
                answer = self.get_updates_for_employee(emp_name, days_ago)
                self.history[-1]['context_types'] = ['updates']
                self.history[-1]['response'] = answer
                self._set_to_cache(cache_key, answer, ttl=7200)
                return answer

            dept_id_pattern = r"department id of\s+(.+)"
            match = re.search(dept_id_pattern, query_lower)
            if match:
                dept_name = match.group(1).strip()
                answer = self.get_department_id(dept_name)
                self.history[-1]['context_types'] = ['employees']
                self.history[-1]['response'] = answer
                self._set_to_cache(cache_key, answer, ttl=86400)
                return answer

            if any(phrase in query_lower for phrase in ["list all departments", "show all departments", "all departments"]):
                answer = self.list_departments()
                self.history[-1]['context_types'] = ['employees']
                self.history[-1]['response'] = answer
                self._set_to_cache(cache_key, answer, ttl=86400)
                return answer
            
            if any(phrase in query_lower for phrase in ["employee count", "how many employees", "total number of employees", "count all employees", "how many people work at eoxs", "total headcount", "staff count", "total staff"]):
                try:
                    response = self.supabase.table('employees').select('employee_id', count='exact').execute()
                    count = response.count
                    answer = f"[EMPLOYEES Context] EOXS currently has **{count} employees** based on the latest data in the database."
                    self.history[-1]['response'] = answer
                    self._set_to_cache(cache_key, answer, ttl=7200)
                    return answer
                except Exception as e:
                    return f"[EMPLOYEES Context] Sorry, I couldn't count employees due to: {str(e)}"

            if any(phrase in query_lower for phrase in ["show all the employees", "all employees of eoxs", "show their names", "all employees name", "total employees name", "print all the members"]):
                alphabetical = "alphabetical" in query_lower
                with_details = "with details" in query_lower or "numbering and details" in query_lower
                names = self.get_employee_names(alphabetical=alphabetical, with_details=with_details)
                if names:
                    if with_details:
                        context = "\n".join(names)
                        intent = "person_info"
                        history_context = "\n".join(f"Q: {h['query']}\nA: {h['response']}" for h in self.history[-4:-1] if h.get('response'))
                        prompt = self.create_chatgpt_style_prompt(context, query_to_use, intent, history_context)
                        gen_response = self.model.generate_content(prompt)
                        answer = f"[EMPLOYEES Context] {gen_response.text}"
                    else:
                        answer = "[EMPLOYEES Context] List of employees:\n" + "\n".join(f"{i+1}. {name}" for i, name in enumerate(names))
                    self.history[-1]['context_types'] = ['employees']
                    self.history[-1]['response'] = answer
                    self._set_to_cache(cache_key, answer, ttl=86400)
                    return answer
                return "[EMPLOYEES Context] Could not retrieve employee names."

            context_types = explicit_contexts or self.determine_context(query_to_use)
            relevant_chunks, used_contexts = self.get_relevant_chunks(query_to_use, context_types)
            self.history[-1]['context_types'] = used_contexts 

            if not relevant_chunks:
                answer = "I don't have enough information to answer this question."
            else:
                context_str = "\n---\n".join(relevant_chunks) 
                intent = self.detect_intent(query_to_use) 
                history_context = "\n".join(f"Q: {h['query']}\nA: {h['response']}" for h in self.history[-4:-1] if h.get('response'))
                prompt = self.create_chatgpt_style_prompt(context_str, query_to_use, intent, history_context)
                gen_response = self.model.generate_content(prompt)
                answer = gen_response.text
            
            self.history[-1]['response'] = answer
            self._set_to_cache(cache_key, answer, ttl=10800)
            return answer

        except Exception as e:
            print(f"An error occurred while answering the query: {e}")
            return f"Sorry, I encountered an error: {e}"

    def clear_milvus_collection(self, collection_name: str):
        if collection_name in self.contexts:
            try:
                if utility.has_collection(collection_name):
                    utility.drop_collection(collection_name)
                    self.contexts[collection_name]['collection'] = None 
                    print(f"Successfully dropped Milvus collection: {collection_name}")
                else:
                    print(f"Milvus collection '{collection_name}' does not exist.")
            except Exception as e:
                print(f"Error dropping Milvus collection '{collection_name}': {str(e)}")
        else:
            print(f"Unknown collection name: {collection_name}. Must be one of {list(self.contexts.keys())}")

    def clear_all_milvus_data(self):
        clear_start_time = time.perf_counter() 
        print("\nWARNING: Attempting to delete ALL configured Milvus collections. This action is irreversible.")
        for col_name in list(self.contexts.keys()): 
            self.clear_milvus_collection(col_name)
        clear_end_time = time.perf_counter() 
        print(f"\nAll configured Zilliz collections have been processed for deletion in {clear_end_time - clear_start_time:.2f} seconds.") 

    def clear_supabase_conversation_history(self):
        try:
            self.supabase.table('conversation_history').delete().gt('id', 0).execute()
            self.history = [] 
            print("Successfully cleared Supabase 'conversation_history'.")
        except Exception as e:
            print(f"Error clearing Supabase conversation history: {str(e)}")

@celery_app.task(name='chatbot.nightly_update_task')
def nightly_update_task():
    print("Executing nightly update task...")
    try:
        chatbot_instance = InteractiveRAGChatbot()
        chatbot_instance.nightly_update_of_daily_updates()
        print("Nightly update task completed successfully.")
    except Exception as e:
        print(f"An error occurred during the nightly update task: {e}")

if __name__ == '__main__':
    chatbot = InteractiveRAGChatbot()
    
    initial_load_start_time = time.perf_counter()
    chatbot.load_all_data() 
    initial_load_end_time = time.perf_counter()
    print(f"Initial data loading completed in {initial_load_end_time - initial_load_start_time:.2f} seconds.")
    
    print("\nWelcome to the Interactive RAG Chatbot!")
    print("Type 'exit' to end the chat.")
    print("Type 'clear_milvus' to clear all Milvus data and reload it.")
    print("\nTo run the nightly update worker, use the following commands in separate terminals:")
    print("1. Start Celery worker: celery -A chatbot.celery_app worker --loglevel=info")
    print("2. Start Celery beat scheduler: celery -A chatbot.celery_app beat --loglevel=info")

    while True:
        user_query = input("\nYour question: ")
        if user_query.lower() == 'exit':
            break
        elif user_query.lower() == 'clear_milvus':
            print("\nInitiating Milvus data clear and reload...")
            clear_and_reload_start_time = time.perf_counter() 
            
            chatbot.clear_all_milvus_data()
            chatbot.load_all_data() 
            
            clear_and_reload_end_time = time.perf_counter() 
            print(f"\nMilvus data cleared and reloaded. Total time for clear_milvus command: {clear_and_reload_end_time - clear_and_reload_start_time:.2f} seconds.")
            print("You can now ask questions based on the refreshed data.")
            continue 
        
        response = chatbot.answer_query(user_query)
        print("\nKYNEX:")
        print(response)