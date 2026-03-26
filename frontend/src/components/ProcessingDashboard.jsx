const STEPS = [
  { label: 'Analyzing video metadata',      threshold: 15 },
  { label: 'Detecting scenes with AI',       threshold: 30 },
  { label: 'Removing silent/blank segments', threshold: 50 },
  { label: 'Applying smart trimming',        threshold: 65 },
  { label: 'Adding transitions',             threshold: 80 },
  { label: 'Synchronizing audio',            threshold: 90 },
  { label: 'Encoding final video',           threshold: 100 },
]

export default function ProcessingDashboard({ jobStatus }) {
  const progress = jobStatus?.progress ?? 0
  const message  = jobStatus?.message  ?? 'Initializing...'

  const getStepStatus = (threshold, idx) => {
    if (progress >= threshold) return 'completed'
    const prevThreshold = idx > 0 ? STEPS[idx - 1].threshold : 0
    if (progress >= prevThreshold && progress < threshold) return 'active'
    return 'pending'
  }

  const getStepIcon = (status) => {
    if (status === 'completed') return '✅'
    if (status === 'active')    return '⚙️'
    return '⏳'
  }

  return (
    <div className="processing-section">
      <div className="processing-header">
        <h2>🤖 AI is editing your video...</h2>
        <p>This may take a few minutes depending on video length</p>
      </div>

      <div className="glass-card">
        <div className="progress-container">
          <div className="progress-bar-wrapper">
            <div className="progress-bar" style={{ width: `${progress}%` }} />
          </div>
          <div className="progress-info">
            <span className="progress-message">{message}</span>
            <span className="progress-percentage">{progress}%</span>
          </div>
        </div>

        <div className="steps-list">
          {STEPS.map((step, idx) => {
            const status = getStepStatus(step.threshold, idx)
            return (
              <div key={step.label} className={`step-item ${status}`}>
                <span className="step-icon">{getStepIcon(status)}</span>
                <span className="step-label">{step.label}</span>
              </div>
            )
          })}
        </div>
      </div>
    </div>
  )
}
