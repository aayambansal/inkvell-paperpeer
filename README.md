# 🏆 PeerReviewer - AI Academic Paper Review System

**Multi-model validation pipeline for comprehensive academic paper evaluation**

PeerReviewer combines multiple state-of-the-art AI models to provide professional-grade academic peer reviews. Built with MT-Bench calibration and robust validation, it delivers unbiased, evidence-based feedback for research papers.

## ✨ Key Features

- **🤖 Multi-Expert AI Panel** - GPT-4o, Claude Sonnet 3.5, GPT-4 Turbo, GPT-4o Mini
- **📊 MT-Bench Calibrated** - Unbiased scoring based on real model performance data  
- **🛡️ Academic Validation** - Strict paper detection, rejects non-academic documents
- **🎨 Professional Interface** - Modern, responsive web application
- **⚡ Fast Processing** - ~60 seconds per paper with real-time updates

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- OpenAI API Key
- Anthropic API Key

### Installation

```bash
# Clone repository
git clone https://github.com/aayambansal/inkvell-paperpeer.git
cd inkvell-paperpeer

# Install dependencies
pip install -r requirements.txt

# Set environment variables
export OPENAI_API_KEY="your-openai-key"
export ANTHROPIC_API_KEY="your-anthropic-key"

# Run application
python app.py
```

Visit `http://localhost:8000` to start reviewing papers.

## 🔧 Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `OPENAI_API_KEY` | OpenAI API key from platform.openai.com | ✅ |
| `ANTHROPIC_API_KEY` | Anthropic API key from console.anthropic.com | ✅ |

## 📁 Project Structure

```
inkvell-paperpeer/
├── src/ultimate_app.py      # Main Flask application
├── templates/index.html     # Web interface
├── static/assets/          # Static files
├── data/mt_bench/          # MT-Bench calibration data
├── uploads/                # PDF upload directory
├── mtbench_calibrator.py   # Scoring calibration system
├── requirements.txt        # Python dependencies
├── vercel.json            # Vercel deployment config
└── app.py                 # Application entry point
```

## 🎯 How It Works

1. **Upload PDF** - Academic papers only (resumes/CVs rejected)
2. **Multi-Model Analysis** - 4 AI experts provide specialized reviews
3. **Consensus Generation** - Weighted scoring with consistency optimization
4. **Professional Output** - Conference-ready feedback and recommendations

## 📊 Scoring System

- **8.5+ Top Tier** - Nature, Science level
- **7.5+ Tier 1** - NeurIPS, ICML, ICLR  
- **6.5+ Tier 2** - EMNLP, ACL, AAAI
- **5.5+ Tier 3** - Strong conferences
- **<5.5 Needs Work** - Major revision required

## 🚀 Deployment

### Vercel (Recommended)
1. Fork this repository
2. Connect to Vercel
3. Add environment variables
4. Deploy automatically

### Local Development
```bash
python app.py
```

## 🛡️ Quality Guarantees

- ✅ **No Hallucination** - Evidence-based analysis only
- ✅ **Academic Focus** - Strict paper validation  
- ✅ **Unbiased Scoring** - MT-Bench calibrated assessment
- ✅ **Professional Output** - Conference-tier review quality

## 🤝 Contributing

1. Fork the repository
2. Create feature branch
3. Submit pull request

## 📜 License

MIT License - see [LICENSE](LICENSE) for details.

---

**Transform your research with AI-powered peer review** 🚀

[⭐ Star on GitHub](https://github.com/aayambansal/inkvell-paperpeer) | [🚀 Deploy on Vercel](https://vercel.com/new)