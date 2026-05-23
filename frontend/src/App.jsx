import React, { useState } from 'react';
import ImageClassifier from './components/ImageClassifier';
import BenchmarkDashboard from './components/BenchmarkDashboard';
import { ImageIcon, BarChart3, Binary } from 'lucide-react';

const App = () => {
  const [activeTab, setActiveTab] = useState('classify');

  return (
    <div className="min-h-screen bg-[#0a0f1e] text-slate-200 font-sans selection:bg-blue-500/30">
      {/* Dynamic Background */}
      <div className="fixed inset-0 overflow-hidden pointer-events-none">
        <div className="absolute top-[-10%] left-[-10%] w-[40%] h-[40%] bg-blue-600/10 rounded-full blur-[120px]"></div>
        <div className="absolute bottom-[-10%] right-[-10%] w-[40%] h-[40%] bg-purple-600/10 rounded-full blur-[120px]"></div>
      </div>

      <div className="relative z-10 container mx-auto px-4 py-8">
        {/* Header */}
        <header className="text-center mb-12">
          <div className="inline-flex items-center justify-center p-3 bg-blue-500/10 rounded-2xl mb-4 border border-blue-500/20">
            <Binary className="text-blue-400" size={32} />
          </div>
          <h1 className="text-4xl md:text-5xl font-black text-white tracking-tighter mb-2">
            CIFAR-10 <span className="text-transparent bg-clip-text bg-gradient-to-r from-blue-400 to-purple-400">Visual Classifier</span>
          </h1>
          <p className="text-slate-400 font-medium tracking-wide uppercase text-xs">
            MLP vs CNN Architecture Benchmark
          </p>
        </header>

        {/* Tab Navigation */}
        <nav className="flex justify-center mb-12">
          <div className="bg-slate-900/50 backdrop-blur-md p-1.5 rounded-2xl border border-white/5 flex gap-1 shadow-2xl">
            <button
              onClick={() => setActiveTab('classify')}
              className={`flex items-center gap-2 px-6 py-3 rounded-xl font-bold transition-all duration-300 ${
                activeTab === 'classify'
                  ? 'bg-blue-600 text-white shadow-lg shadow-blue-600/20'
                  : 'text-slate-400 hover:text-white hover:bg-white/5'
              }`}
            >
              <ImageIcon size={18} />
              <span>Classify Image</span>
            </button>
            <button
              onClick={() => setActiveTab('benchmark')}
              className={`flex items-center gap-2 px-6 py-3 rounded-xl font-bold transition-all duration-300 ${
                activeTab === 'benchmark'
                  ? 'bg-blue-600 text-white shadow-lg shadow-blue-600/20'
                  : 'text-slate-400 hover:text-white hover:bg-white/5'
              }`}
            >
              <BarChart3 size={18} />
              <span>Model Benchmark</span>
            </button>
          </div>
        </nav>

        {/* Main Content */}
        <main className="animate-in fade-in duration-700">
          {activeTab === 'classify' ? (
            <ImageClassifier />
          ) : (
            <BenchmarkDashboard />
          )}
        </main>

        {/* Footer */}
        <footer className="mt-20 py-8 border-t border-white/5 text-center">
          <p className="text-slate-600 text-sm font-medium">
            &copy; {new Date().getFullYear()} CIFAR-10 Visual Classifier. Built with FastAPI, TensorFlow & React.
          </p>
        </footer>
      </div>
    </div>
  );
};

export default App;
