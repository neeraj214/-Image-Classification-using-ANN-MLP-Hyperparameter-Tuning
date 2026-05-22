import React, { useState, useEffect } from 'react';
import { Upload, Image as ImageIcon, CheckCircle, AlertCircle, BarChart3, Info } from 'lucide-react';
import { predictImage, getBenchmark, checkHealth } from './api/client';

function App() {
  const [selectedFile, setSelectedFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [results, setResults] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [benchmark, setBenchmark] = useState(null);
  const [apiStatus, setApiStatus] = useState('connecting');

  useEffect(() => {
    const fetchInitialData = async () => {
      try {
        const health = await checkHealth();
        setApiStatus(health.status === 'ok' ? 'online' : 'error');
        
        const benchData = await getBenchmark();
        setBenchmark(benchData);
      } catch (err) {
        console.error('Initial fetch failed:', err);
        setApiStatus('offline');
      }
    };
    fetchInitialData();
  }, []);

  const handleFileChange = (e) => {
    const file = e.target.files[0];
    if (file) {
      setSelectedFile(file);
      setPreview(URL.createObjectURL(file));
      setResults(null);
      setError(null);
    }
  };

  const handleUpload = async () => {
    if (!selectedFile) return;

    setLoading(true);
    setError(null);
    try {
      const data = await predictImage(selectedFile);
      setResults(data);
    } catch (err) {
      setError(err.response?.data?.detail || 'Failed to classify image');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen p-4 md:p-8 max-w-6xl mx-auto">
      <header className="mb-8 flex justify-between items-center border-b pb-4">
        <div>
          <h1 className="text-3xl font-bold text-slate-800">CIFAR-10 Visual Classifier</h1>
          <p className="text-slate-500">Benchmark: MLP vs CNN Architectures</p>
        </div>
        <div className="flex items-center gap-2">
          <span className={`w-3 h-3 rounded-full ${
            apiStatus === 'online' ? 'bg-green-500' : 
            apiStatus === 'offline' ? 'bg-red-500' : 'bg-yellow-500 animate-pulse'
          }`}></span>
          <span className="text-sm font-medium uppercase tracking-wider text-slate-600">
            API: {apiStatus}
          </span>
        </div>
      </header>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
        {/* Classification Section */}
        <div className="lg:col-span-2 space-y-6">
          <section className="bg-white p-6 rounded-2xl shadow-sm border border-slate-200">
            <h2 className="text-xl font-semibold mb-4 flex items-center gap-2">
              <ImageIcon className="text-primary" /> Image Classification
            </h2>
            
            <div className="flex flex-col items-center justify-center border-2 border-dashed border-slate-300 rounded-xl p-8 transition-colors hover:border-primary">
              {preview ? (
                <div className="relative group w-48 h-48 mb-4">
                  <img src={preview} alt="Preview" className="w-full h-full object-cover rounded-lg shadow-md" />
                  <button 
                    onClick={() => {setSelectedFile(null); setPreview(null); setResults(null);}}
                    className="absolute -top-2 -right-2 bg-red-500 text-white p-1 rounded-full opacity-0 group-hover:opacity-100 transition-opacity"
                  >
                    <AlertCircle size={16} />
                  </button>
                </div>
              ) : (
                <label className="flex flex-col items-center cursor-pointer">
                  <Upload size={48} className="text-slate-400 mb-2" />
                  <span className="text-slate-600 font-medium">Click to upload image</span>
                  <span className="text-slate-400 text-sm">(CIFAR-10 categories only)</span>
                  <input type="file" className="hidden" onChange={handleFileChange} accept="image/*" />
                </label>
              )}

              <button 
                onClick={handleUpload}
                disabled={!selectedFile || loading || apiStatus !== 'online'}
                className={`mt-4 px-8 py-3 rounded-full font-semibold text-white transition-all shadow-md ${
                  !selectedFile || loading || apiStatus !== 'online' 
                  ? 'bg-slate-300 cursor-not-allowed' 
                  : 'bg-primary hover:bg-blue-600 active:scale-95'
                }`}
              >
                {loading ? 'Classifying...' : 'Analyze Image'}
              </button>
            </div>

            {error && (
              <div className="mt-4 p-4 bg-red-50 text-red-700 rounded-lg flex items-center gap-2 border border-red-100">
                <AlertCircle size={20} />
                <span>{error}</span>
              </div>
            )}
          </section>

          {/* Results Comparison */}
          {results && (
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6 animate-in fade-in slide-in-from-bottom-4 duration-500">
              {['cnn', 'mlp'].map((model) => (
                <div key={model} className="bg-white p-6 rounded-2xl shadow-sm border border-slate-200">
                  <div className="flex justify-between items-center mb-4">
                    <h3 className="text-lg font-bold uppercase tracking-wider text-slate-700">{model} Result</h3>
                    <CheckCircle className={model === 'cnn' ? 'text-secondary' : 'text-primary'} />
                  </div>
                  <div className="space-y-4">
                    <div className="flex justify-between items-end">
                      <span className="text-4xl font-black capitalize text-slate-900">{results[model].class}</span>
                      <span className="text-slate-500 font-medium">{(results[model].confidence * 100).toFixed(1)}% match</span>
                    </div>
                    <div className="w-full bg-slate-100 rounded-full h-3">
                      <div 
                        className={`h-3 rounded-full transition-all duration-1000 ${model === 'cnn' ? 'bg-secondary' : 'bg-primary'}`}
                        style={{ width: `${results[model].confidence * 100}%` }}
                      ></div>
                    </div>
                    <div className="pt-4 border-t border-slate-50">
                      <h4 className="text-xs font-bold text-slate-400 uppercase mb-2">Top Probabilities</h4>
                      <div className="space-y-1">
                        {Object.entries(results[model].probabilities)
                          .sort(([,a], [,b]) => b - a)
                          .slice(0, 3)
                          .map(([name, prob]) => (
                            <div key={name} className="flex justify-between text-sm">
                              <span className="capitalize text-slate-600">{name}</span>
                              <span className="font-mono text-slate-500">{(prob * 100).toFixed(1)}%</span>
                            </div>
                          ))}
                      </div>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>

        {/* Sidebar / Benchmarks */}
        <div className="space-y-6">
          <section className="bg-white p-6 rounded-2xl shadow-sm border border-slate-200">
            <h2 className="text-xl font-semibold mb-4 flex items-center gap-2">
              <BarChart3 className="text-accent" /> Model Benchmarks
            </h2>
            {benchmark ? (
              <div className="space-y-4">
                {Object.entries(benchmark).map(([name, stats]) => (
                  <div key={name} className="p-3 bg-slate-50 rounded-xl border border-slate-100">
                    <h3 className="text-xs font-bold text-slate-400 uppercase mb-2 truncate">{name.replace('_', ' ')}</h3>
                    <div className="grid grid-cols-2 gap-2 text-sm">
                      <div className="flex flex-col">
                        <span className="text-slate-500 text-[10px] uppercase font-bold">Accuracy</span>
                        <span className="font-bold text-slate-800">{(stats.accuracy * 100).toFixed(1)}%</span>
                      </div>
                      <div className="flex flex-col">
                        <span className="text-slate-500 text-[10px] uppercase font-bold">Train Time</span>
                        <span className="font-medium text-slate-800">{stats.training_time?.toFixed(1)}s</span>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            ) : (
              <div className="py-8 text-center text-slate-400">
                <Info size={24} className="mx-auto mb-2 opacity-50" />
                <p className="text-sm">Run benchmark script to see results</p>
              </div>
            )}
          </section>

          <section className="bg-blue-600 p-6 rounded-2xl shadow-lg text-white">
            <h3 className="text-lg font-bold mb-2">Project Info</h3>
            <p className="text-blue-100 text-sm mb-4">
              Comparing deep neural networks on the CIFAR-10 image dataset (60,000 32x32 color images in 10 classes).
            </p>
            <div className="space-y-2">
              <div className="flex justify-between text-xs font-bold text-blue-200 uppercase">
                <span>Classes</span>
                <span>10</span>
              </div>
              <div className="flex justify-between text-xs font-bold text-blue-200 uppercase">
                <span>Dataset</span>
                <span>60k Images</span>
              </div>
            </div>
          </section>
        </div>
      </div>
    </div>
  );
}

export default App;
