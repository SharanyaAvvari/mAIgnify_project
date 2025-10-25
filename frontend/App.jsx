import React, { useState } from 'react';

function App() {
  const [prompt, setPrompt] = useState('');
  const [response, setResponse] = useState('');

  const handleSubmit = async () => {
    setResponse('Running...');
    const res = await fetch('http://localhost:8000/run-prompt', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ prompt })
    });
    const data = await res.json();
    setResponse(JSON.stringify(data, null, 2));
  };

  return (
    <div style={{ padding: 20 }}>
      <h1>mAIstro Prompt Runner</h1>
      <textarea
        rows={7}
        cols={80}
        placeholder="Paste your prompt here"
        value={prompt}
        onChange={e => setPrompt(e.target.value)}
      />
      <br />
      <button onClick={handleSubmit}>Run Prompt</button>
      <pre>{response}</pre>
    </div>
  );
}

export default App;
