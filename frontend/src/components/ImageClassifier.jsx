import React, { useState, useCallback } from 'react';
import { predictImage } from '../api/cifarApi';
import { Upload, Loader2, Image as ImageIcon, CheckCircle2 } from 'lucide-react';

const classNames = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'];

const ImageClassifier = () => {
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [results, setResults] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const onFileChange = (e) => {
    const selectedFile = e.target.files[0];
    if (selectedFile) {
      processFile(selectedFile);
    }
  };

  const processFile = (selectedFile) => {
    setFile(selectedFile);
    setPreview(URL.createObjectURL(selectedFile));
    setResults(null);
    setError(null);
    handlePredict(selectedFile);
  };

  const handlePredict = async (selectedFile) => {
    setLoading(true);
    try {
      const data = await predictImage(selectedFile);
      setResults(data);
    } catch (err) {
      setError(err.message || 'Classification failed');
    } finally {
      setLoading(false);
    }
  };

  const ResultCard = ({ title, result, isWinner }) => (
    <div className={`flex-1 p-6 rounded-2xl bg-white/10 backdrop-blur-md border ${isWinner ? 'border-green-500/50 shadow-[0_0_15px_rgba(34,197,94,0.3)]' : 'border-white/20'} transition-all duration-500`}>
      <div className="flex justify-between items-center mb-4">
        <h3 className="text-xl font-bold text-white uppercase tracking-wider">{title}</h3>
        {isWinner && <CheckCircle2 className="text-green-500" size={20} />}
      </div>
      
      <div className="mb-6">
        <span className="text-4xl font-black text-white capitalize block mb-1">{result.class}</span>
        <span className="text-sm font-medium text-slate-300">{(result.confidence * 100).toFixed(1)}% Confidence</span>
      </div>

      <div className="space-y-3">
        <h4 className="text-xs font-bold text-slate-400 uppercase tracking-widest">Top Predictions</h4>
        {Object.entries(result.probabilities)
          .sort(([, a], [, b]) => b - a)
          .slice(0, 3)
          .map(([name, prob]) => (
            <div key={name} className="space-y-1">
              <div className="flex justify-between text-xs text-slate-200 capitalize">
                <span>{name}</span>
                <span>{(prob * 100).toFixed(1)}%</span>
              </div>
              <div className="w-full bg-white/5 rounded-full h-1.5 overflow-hidden">
                <div 
                  className={`h-full transition-all duration-1000 ${isWinner ? 'bg-green-500' : 'bg-blue-500'}`}
                  style={{ width: `${prob * 100}%` }}
                ></div>
              </div>
            </div>
          ))}
      </div>
    </div>
  );

  return (
    <div className="max-w-5xl mx-auto space-y-8 p-4">
      {/* Upload Section */}
      <section className="relative group">
        <div className="absolute -inset-1 bg-gradient-to-r from-blue-600 to-purple-600 rounded-3xl blur opacity-25 group-hover:opacity-50 transition duration-1000 group-hover:duration-200"></div>
        <div className="relative bg-slate-900/80 backdrop-blur-xl p-8 rounded-3xl border border-white/10 shadow-2xl">
          <div className="flex flex-col md:flex-row items-center gap-8">
            {/* Preview Box */}
            <div className="w-32 h-32 rounded-2xl border-2 border-dashed border-white/20 flex items-center justify-center overflow-hidden bg-white/5 group/preview shrink-0">
              {preview ? (
                <img src={preview} alt="Preview" className="w-full h-full object-cover" />
              ) : (
                <ImageIcon className="text-white/20 group-hover/preview:text-white/40 transition-colors" size={48} />
              )}
            </div>

            {/* Dropzone */}
            <label className="flex-1 w-full cursor-pointer group/zone">
              <div className="border-2 border-dashed border-white/10 group-hover/zone:border-blue-500/50 rounded-2xl p-8 text-center transition-all bg-white/5 hover:bg-white/10">
                <Upload className="mx-auto mb-4 text-slate-400 group-hover/zone:text-blue-400 transition-colors" size={32} />
                <p className="text-slate-300 font-medium mb-1">Drag and drop or click to upload</p>
                <p className="text-slate-500 text-xs">Supports: JPG, PNG, WEBP (CIFAR-10 Categories)</p>
                <input type="file" className="hidden" onChange={onFileChange} accept="image/*" />
              </div>
            </label>
          </div>

          {loading && (
            <div className="mt-6 flex items-center justify-center gap-3 text-blue-400 font-bold animate-pulse">
              <Loader2 className="animate-spin" />
              <span>Analyzing with dual models...</span>
            </div>
          )}

          {error && (
            <div className="mt-6 p-4 bg-red-500/10 border border-red-500/50 rounded-xl text-red-400 text-sm text-center">
              {error}
            </div>
          )}
        </div>
      </section>

      {/* Results Section */}
      {results && (
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6 animate-in fade-in slide-in-from-bottom-8 duration-700">
          <ResultCard 
            title="CNN Baseline" 
            result={results.cnn} 
            isWinner={results.cnn.confidence >= results.mlp.confidence} 
          />
          <ResultCard 
            title="MLP Baseline" 
            result={results.mlp} 
            isWinner={results.mlp.confidence > results.cnn.confidence} 
          />
        </div>
      )}

      {/* Static Reference Grid */}
      <section className="bg-slate-900/50 backdrop-blur-md p-6 rounded-3xl border border-white/5">
        <h3 className="text-xs font-bold text-slate-500 uppercase tracking-widest mb-4">CIFAR-10 Category Reference</h3>
        <div className="grid grid-cols-5 md:grid-cols-10 gap-3">
          {classNames.map((name) => (
            <div key={name} className="flex flex-col items-center gap-2">
              <div className="w-full aspect-square bg-white/5 rounded-lg border border-white/10 flex items-center justify-center">
                <span className="text-[10px] text-slate-600 font-black uppercase tracking-tighter">{name.slice(0, 3)}</span>
              </div>
              <span className="text-[10px] text-slate-400 font-medium capitalize">{name}</span>
            </div>
          ))}
        </div>
      </section>
    </div>
  );
};

export default ImageClassifier;
