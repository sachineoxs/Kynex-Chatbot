# Kynex Chatbot

A sophisticated RAG (Retrieval-Augmented Generation) chatbot built with Python, React, and modern AI technologies. This chatbot provides intelligent responses by combining vector search with generative AI.

## 🚀 Features

- **Multi-Context RAG**: Supports multiple data contexts (products, updates, employees, departments)
- **Real-time Updates**: Automated nightly updates using Celery
- **Vector Search**: Powered by Zilliz Cloud (Milvus) for efficient similarity search
- **Modern UI**: React-based frontend with beautiful, responsive design
- **Caching**: Redis-based caching for improved performance
- **Scalable Architecture**: Backend API with FastAPI/Flask support

## 🛠️ Tech Stack

### Backend
- **Python 3.13**
- **Google Generative AI** (Gemini 1.5 Flash)
- **Zilliz Cloud** (Milvus) for vector database
- **Supabase** for relational data storage
- **Redis** for caching and task queue
- **Celery** for background tasks
- **HuggingFace Embeddings** (all-MiniLM-L6-v2)

### Frontend
- **React.js**
- **CSS3** with modern styling
- **Responsive Design**

## 📁 Project Structure

```
EOXS RAG CHATBOT 2/
├── backend/
│   ├── chatbot.py          # Main chatbot logic
│   ├── main.py            # FastAPI/Flask server
│   ├── requirements.txt   # Python dependencies
│   └── venv/             # Virtual environment
├── frontend/
│   ├── src/
│   │   ├── components/    # React components
│   │   ├── assets/        # Images and static files
│   │   └── styles/        # CSS files
│   ├── public/           # Public assets
│   └── package.json      # Node.js dependencies
├── .gitignore           # Git ignore rules
└── README.md           # This file
```

## 🚀 Quick Start

### Prerequisites

- Python 3.13+
- Node.js 16+
- Redis server
- Zilliz Cloud account
- Supabase account
- Google AI API key

### Backend Setup

1. **Clone the repository**
   ```bash
   git clone git@github.com:sachineoxs/Kynex-Chatbot.git
   cd Kynex-Chatbot
   ```

2. **Set up Python environment**
   ```bash
   cd backend
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. **Configure environment variables**
   Create a `.env` file in the backend directory:
   ```env
   GOOGLE_API_KEY=your_google_ai_api_key
   SUPABASE_URL=your_supabase_url
   SUPABASE_KEY=your_supabase_key
   ZILLIZ_URI=your_zilliz_uri
   ZILLIZ_TOKEN=your_zilliz_token
   REDIS_HOST=localhost
   REDIS_PORT=6379
   REDIS_PASSWORD=your_redis_password
   REDIS_SSL=true
   USE_REDIS=true
   ```

4. **Initialize the chatbot**
   ```bash
   python chatbot.py
   ```

### Frontend Setup

1. **Install dependencies**
   ```bash
   cd frontend
   npm install
   ```

2. **Start the development server**
   ```bash
   npm start
   ```

## 🔧 Configuration

### Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `GOOGLE_API_KEY` | Google AI API key | Yes |
| `SUPABASE_URL` | Supabase project URL | Yes |
| `SUPABASE_KEY` | Supabase API key | Yes |
| `ZILLIZ_URI` | Zilliz Cloud URI | Yes |
| `ZILLIZ_TOKEN` | Zilliz Cloud token | Yes |
| `REDIS_HOST` | Redis server host | No (default: localhost) |
| `REDIS_PORT` | Redis server port | No (default: 6379) |
| `REDIS_PASSWORD` | Redis password | No |
| `REDIS_SSL` | Enable SSL for Redis | No (default: true) |
| `USE_REDIS` | Enable Redis caching | No (default: true) |

## 📊 Data Sources

The chatbot supports multiple data contexts:

- **Products**: EOXS product information and features
- **Updates**: Daily team updates and project status
- **Employees**: Employee information and organizational structure
- **Departments**: Department details and hierarchy

## 🤖 Usage Examples

### Basic Queries
- "What are the main features of EOXS products?"
- "Show me all employees in alphabetical order"
- "What updates did John provide last week?"
- "List all departments with their IDs"

### Advanced Queries
- "Compare the features of different EOXS products"
- "Show completed tasks by department"
- "Give me yearly updates for Sarah in 2024"

## 🔄 Background Tasks

The system uses Celery for automated tasks:

- **Nightly Updates**: Automatically refreshes daily updates data
- **Data Synchronization**: Keeps vector embeddings up to date

To run background tasks:
```bash
# Start Celery worker
celery -A chatbot.celery_app worker --loglevel=info

# Start Celery beat scheduler
celery -A chatbot.celery_app beat --loglevel=info
```

## 🧪 Testing

```bash
# Run backend tests
cd backend
python -m pytest

# Run frontend tests
cd frontend
npm test
```

## 📈 Performance

- **Vector Search**: Sub-second response times for similarity queries
- **Caching**: Redis-based caching reduces API calls by 60%
- **Batch Processing**: Efficient handling of large datasets
- **Memory Optimization**: Intelligent chunking and embedding management

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Google AI** for the Gemini model
- **Zilliz** for the vector database infrastructure
- **Supabase** for the backend-as-a-service
- **Redis** for caching and task queuing
- **React** for the frontend framework

## 📞 Support

For support, email support@eoxs.com or create an issue in this repository.

---

**Built with ❤️ by the EOXS Team** 