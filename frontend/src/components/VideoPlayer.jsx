export default function VideoPlayer({ jobId, jobStatus, onReset }) {
  const handleDownload = () => {
    const link = document.createElement('a')
    link.href = `/api/download/${jobId}`
    link.download = jobStatus?.result_filename || 'edited-video.mp4'
    document.body.appendChild(link)
    link.click()
    document.body.removeChild(link)
  }

  const formatDate = (iso) => {
    if (!iso) return 'N/A'
    return new Date(iso).toLocaleString()
  }

  return (
    <div className="player-section">
      <div className="player-header">
        <h2>✅ Your video is ready!</h2>
        <p>Your video has been intelligently edited by AI</p>
      </div>

      <div className="glass-card">
        <div className="stats-card">
          <div className="stat-row">
            <span className="stat-label">Status</span>
            <span className="stat-value success">✅ Completed</span>
          </div>
          <div className="stat-row">
            <span className="stat-label">Output File</span>
            <span className="stat-value">{jobStatus?.result_filename || 'edited-video.mp4'}</span>
          </div>
          <div className="stat-row">
            <span className="stat-label">Job ID</span>
            <span className="stat-value" style={{ fontSize: '0.75rem', opacity: 0.7 }}>{jobId}</span>
          </div>
          <div className="stat-row">
            <span className="stat-label">Completed At</span>
            <span className="stat-value">{formatDate(jobStatus?.created_at)}</span>
          </div>
          <div className="stat-row">
            <span className="stat-label">Processing</span>
            <span className="stat-value success">100% Complete</span>
          </div>
        </div>

        <div className="player-actions">
          <button className="btn-primary" onClick={handleDownload}>
            ⬇️ Download Edited Video
          </button>
          <button className="btn-secondary" onClick={onReset}>
            🔄 Edit Another Video
          </button>
        </div>
      </div>
    </div>
  )
}
