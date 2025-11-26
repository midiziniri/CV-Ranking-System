// static/js/ai_features.js
class AIFeatures {
    constructor() {
        this.initializeDashboard();
    }
    
    async loadAIInsights(jobId) {
        try {
            const response = await fetch(`/HR1/ai_insights/${jobId}`);
            const data = await response.json();
            this.updateInsightsDisplay(data);
        } catch (error) {
            console.error('Failed to load AI insights:', error);
        }
    }
    
    updateInsightsDisplay(data) {
        const mlStatus = document.getElementById('ml-status');
        if (data.system_insights.model_trained) {
            mlStatus.textContent = '✅ Ready';
            mlStatus.className = 'metric-value success';
        } else {
            mlStatus.textContent = '🔄 Training';
            mlStatus.className = 'metric-value warning';
        }
    }
    
    async submitFeedback(applicationId, rating, hired) {
        try {
            const formData = new FormData();
            formData.append('application_id', applicationId);
            formData.append('rating', rating);
            formData.append('hired', hired);
            
            const response = await fetch('/HR1/provide_feedback', {
                method: 'POST',
                body: formData
            });
            
            const result = await response.json();
            if (result.message) {
                alert('✅ ' + result.message);
                this.showAILearningAnimation();
            }
        } catch (error) {
            console.error('Feedback submission failed:', error);
            alert('❌ Failed to submit feedback');
        }
    }
    
    showAILearningAnimation() {
        // Visual feedback that AI is learning
        const aiIcon = document.querySelector('.ai-insights-panel h3');
        aiIcon.style.animation = 'pulse 2s ease-in-out';
        setTimeout(() => {
            aiIcon.style.animation = '';
        }, 2000);
    }
}

// Initialize AI features
document.addEventListener('DOMContentLoaded', () => {
    const aiFeatures = new AIFeatures();
    
    // Load AI insights for current job
    const jobId = document.querySelector('[data-job-id]')?.dataset.jobId;
    if (jobId) {
        aiFeatures.loadAIInsights(jobId);
    }
});