import { useState } from 'react'

export default function FadeImage({ className = '', onLoad, ...props }) {
  const [loaded, setLoaded] = useState(false)

  return (
    <img
      {...props}
      className={`fade-img${loaded ? ' loaded' : ''}${className ? ' ' + className : ''}`}
      onLoad={(e) => {
        setLoaded(true)
        onLoad?.(e)
      }}
    />
  )
}
