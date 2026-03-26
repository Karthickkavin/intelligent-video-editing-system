import { useRef, useState } from 'react'

const MAX_SIZE = 500 * 1024 * 1024 // 500MB
const ALLOWED_EXT = ['.mp4', '.mov', '.avi', '.mkv', '.webm']

export default function UploadSection({ onUpload, uploading }) {
  const inputRef = useRef(null)
  const [dragOver, setDragOver] = useState(false)
  const [fileError, setFileError] = useState('')

  const validateFile = (file) => {
    if (!file) return 'No file selected'
    const ext = '.' + file.name.split('.').pop().toLowerCase()
    if (!ALLOWED_EXT.includes(ext)) {
      return `Unsupported format. Please use: ${ALLOWED_EXT.join(', ')}`
    }
    if (file.size > MAX_SIZE) {
      return 'File too large. Maximum size is 500MB.'
    }
    return null
  }

  const handleFile = (file) => {
    const error = validateFile(file)
    if (error) {
      setFileError(error)
      return
    }
    setFileError('')
    onUpload(file)
  }

  const handleClick = () => {
    if (!uploading) inputRef.current?.click()
  }

  const handleChange = (e) => {
    const file = e.target.files?.[0]
    if (file) handleFile(file)
  }

  const handleDrop = (e) => {
    e.preventDefault()
    setDragOver(false)
    const file = e.dataTransfer.files?.[0]
    if (file) handleFile(file)
  }

  const handleDragOver = (e) => {
    e.preventDefault()
    setDragOver(true)
  }

  const handleDragLeave = () => setDragOver(false)

  return (
    <div className="upload-section">
      <div className="upload-title">
        <h1>Transform Your Videos</h1>
        <p>Upload a video and let AI automatically edit it for you</p>
      </div>

      <div className="glass-card">
        {uploading ? (
          <div className="uploading-state">
            <div className="spinner"></div>
            <p>Uploading your video...</p>
          </div>
        ) : (
          <>
            <div
              className={`upload-zone${dragOver ? ' drag-over' : ''}`}
              onClick={handleClick}
              onDrop={handleDrop}
              onDragOver={handleDragOver}
              onDragLeave={handleDragLeave}
            >
              <span className="upload-icon">📁</span>
              <h3>Drop your video here or click to browse</h3>
              <p>Drag &amp; drop your video file here</p>
              <p>or click anywhere in this area</p>
              <div className="upload-formats">
                {['MP4', 'MOV', 'AVI', 'MKV', 'WebM'].map(f => (
                  <span key={f} className="format-badge">{f}</span>
                ))}
              </div>
              <p style={{ marginTop: '1rem', fontSize: '0.8rem', color: '#718096' }}>
                Maximum file size: 500MB
              </p>
            </div>
            {fileError && (
              <p style={{ color: '#fca5a5', marginTop: '1rem', textAlign: 'center', fontSize: '0.9rem' }}>
                ⚠️ {fileError}
              </p>
            )}
          </>
        )}
        <input
          ref={inputRef}
          type="file"
          accept=".mp4,.mov,.avi,.mkv,.webm,video/*"
          style={{ display: 'none' }}
          onChange={handleChange}
        />
      </div>
    </div>
  )
}
