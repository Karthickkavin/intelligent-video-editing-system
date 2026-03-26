import { useState, useEffect, useRef } from 'react'
import axios from 'axios'
import Header from './components/Header'
import UploadSection from './components/UploadSection'
import ProcessingDashboard from './components/ProcessingDashboard'
import VideoPlayer from './components/VideoPlayer'
import './App.css'

function App() {
  const [uploadState, setUploadState] = useState('idle') // idle | uploading | processing | completed | failed
  const [jobId, setJobId] = useState(null)
  const [jobStatus, setJobStatus] = useState(null)
  const [errorMessage, setErrorMessage] = useState('')
  const pollingRef = useRef(null)

  useEffect(() => {
    if (uploadState === 'processing' && jobId) {
      pollingRef.current = setInterval(async () => {
        try {
          const res = await axios.get(`/api/status/${jobId}`)
          setJobStatus(res.data)
          if (res.data.status === 'completed') {
            setUploadState('completed')
            clearInterval(pollingRef.current)
          } else if (res.data.status === 'failed') {
            setUploadState('failed')
            setErrorMessage(res.data.message || 'Processing failed')
            clearInterval(pollingRef.current)
          }
        } catch (err) {
          console.error('Polling error:', err)
        }
      }, 2000)
    }
    return () => {
      if (pollingRef.current) clearInterval(pollingRef.current)
    }
  }, [uploadState, jobId])

  const handleUpload = async (file) => {
    setUploadState('uploading')
    setErrorMessage('')
    try {
      const formData = new FormData()
      formData.append('video', file)
      const res = await axios.post('/api/upload', formData, {
        headers: { 'Content-Type': 'multipart/form-data' }
      })
      setJobId(res.data.job_id)
      setUploadState('processing')
    } catch (err) {
      const msg = err.response?.data?.error || 'Upload failed. Please try again.'
      setErrorMessage(msg)
      setUploadState('failed')
    }
  }

  const handleReset = () => {
    setUploadState('idle')
    setJobId(null)
    setJobStatus(null)
    setErrorMessage('')
    if (pollingRef.current) clearInterval(pollingRef.current)
  }

  return (
    <div className="app">
      <Header />
      <main className="main-content">
        {(uploadState === 'idle' || uploadState === 'uploading') && (
          <UploadSection onUpload={handleUpload} uploading={uploadState === 'uploading'} />
        )}
        {uploadState === 'processing' && (
          <ProcessingDashboard jobStatus={jobStatus} />
        )}
        {uploadState === 'completed' && (
          <VideoPlayer jobId={jobId} jobStatus={jobStatus} onReset={handleReset} />
        )}
        {uploadState === 'failed' && (
          <div className="error-state">
            <div className="error-card">
              <div className="error-icon">❌</div>
              <h2>Processing Failed</h2>
              <p className="error-message">{errorMessage}</p>
              <button className="btn-primary" onClick={handleReset}>
                Try Again
              </button>
            </div>
          </div>
        )}
      </main>
    </div>
  )
}

export default App
