import React, { useState } from 'react';
import '../styles/DailyUpdateForm.css';

const DailyUpdateForm = () => {
  const [formData, setFormData] = useState({
    date: new Date().toISOString().split('T')[0],
    team: '',
    sub_team: '',
    project: 'Chatbot',
    present_members: '',
    summary: '',
    tasks_completed: '',
    blockers: '',
    next_steps: ''
  });

  const handleSubmit = async (e) => {
    e.preventDefault();
    try {
      // Format the data properly
      const dataToSend = {
        ...formData,
        present_members: formData.present_members.split(',').map(m => m.trim()).filter(m => m),
        tasks_completed: formData.tasks_completed.split(',').map(t => t.trim()).filter(t => t),
        blockers: formData.blockers.split(',').map(b => b.trim()).filter(b => b)
      };

      const response = await fetch('http://localhost:8000/api/daily-update', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(dataToSend),
      });
      
      const data = await response.json();
      if (response.ok) {
        alert('Daily update added successfully!');
        // Reset form while keeping the date and project
        setFormData({
          date: formData.date,
          team: '',
          sub_team: '',
          project: 'Chatbot',
          present_members: '',
          summary: '',
          tasks_completed: '',
          blockers: '',
          next_steps: ''
        });
      } else {
        alert(`Error: ${data.error || 'Failed to add update'}`);
      }
    } catch (error) {
      alert('Error submitting update. Please check your connection and try again.');
      console.error('Error:', error);
    }
  };

  return (
    <div className="daily-update-container">
      <div className="daily-update-form">
        <div className="form-header">
          <h2>Daily Update</h2>
          <p className="subtitle">Track your team's progress</p>
        </div>

        <form onSubmit={handleSubmit}>
          <div className="form-section">
            <div className="form-group">
              <label>Date</label>
              <input
                type="date"
                value={formData.date}
                onChange={(e) => setFormData({...formData, date: e.target.value})}
                required
              />
            </div>

            <div className="form-row">
              <div className="form-group">
                <label>Team Name</label>
                <input
                  type="text"
                  value={formData.team}
                  onChange={(e) => setFormData({...formData, team: e.target.value})}
                  placeholder="e.g., AI Generalist"
                  required
                />
              </div>

              <div className="form-group">
                <label>Sub Team</label>
                <input
                  type="text"
                  value={formData.sub_team}
                  onChange={(e) => setFormData({...formData, sub_team: e.target.value})}
                  placeholder="e.g., AI 1"
                  required
                />
              </div>
            </div>

            <div className="form-group">
              <label>Team Members Present</label>
              <div className="input-hint">Separate names with commas</div>
              <input
                type="text"
                value={formData.present_members}
                onChange={(e) => setFormData({...formData, present_members: e.target.value})}
                placeholder="e.g., John Doe, Jane Smith"
                required
              />
            </div>

            <div className="form-group">
              <label>Summary</label>
              <textarea
                value={formData.summary}
                onChange={(e) => setFormData({...formData, summary: e.target.value})}
                placeholder="Brief summary of today's work"
                required
              />
            </div>

            <div className="form-group">
              <label>Tasks Completed</label>
              <div className="input-hint">Separate tasks with commas</div>
              <textarea
                value={formData.tasks_completed}
                onChange={(e) => setFormData({...formData, tasks_completed: e.target.value})}
                placeholder="e.g., Implemented auth, Added tests"
                required
              />
            </div>

            <div className="form-group">
              <label>Blockers</label>
              <div className="input-hint">Separate blockers with commas (if any)</div>
              <textarea
                value={formData.blockers}
                onChange={(e) => setFormData({...formData, blockers: e.target.value})}
                placeholder="e.g., API rate limits, Database connectivity"
              />
            </div>

            <div className="form-group">
              <label>Next Steps</label>
              <textarea
                value={formData.next_steps}
                onChange={(e) => setFormData({...formData, next_steps: e.target.value})}
                placeholder="e.g., Implement user authentication"
                required
              />
            </div>
          </div>

          <div className="form-actions">
            <button type="submit" className="submit-btn">
              Submit Update
              <svg viewBox="0 0 24 24" fill="none" stroke="currentColor">
                <path d="M5 12h14M12 5l7 7-7 7" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
              </svg>
            </button>
          </div>
        </form>
      </div>
    </div>
  );
};

export default DailyUpdateForm; 