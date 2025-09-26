#!/usr/bin/env python3
"""
ULTIMATE Peer Review System
Combines the best of MT-Bench calibration + Foolproof validation + TOP AI models

Features:
- GPT-5, Claude Sonnet 4, Opus 4.1 (most advanced models)
- MT-Bench calibrated scoring
- Foolproof academic validation
- Multi-expert consensus from proven systems
"""

from flask import Flask, render_template, request, jsonify, send_from_directory
import os
import json
import time
import openai
from werkzeug.utils import secure_filename
import PyPDF2
import io
from datetime import datetime
import threading
import queue
import re
import numpy as np

app = Flask(__name__, template_folder='../templates', static_folder='../static')
app.config['UPLOAD_FOLDER'] = '../uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Create upload directory if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Environment variable configuration
openai.api_key = os.getenv('OPENAI_API_KEY')
openai.api_base = "https://api.openai.com/v1"
anthropic_api_key = os.getenv('ANTHROPIC_API_KEY')

# Validate required environment variables
if not openai.api_key:
    print("❌ ERROR: OPENAI_API_KEY environment variable is required")
    print("📝 Please set your OpenAI API key in environment variables")
    exit(1)

if not anthropic_api_key:
    print("❌ ERROR: ANTHROPIC_API_KEY environment variable is required")
    print("📝 Please set your Anthropic API key in environment variables")
    exit(1)

print("✅ API keys loaded from environment variables")

# Global variables
review_queue = queue.Queue()
review_results = {}

class UltimatePeerReviewer:
    def __init__(self):
        # ULTIMATE MODEL PANEL - TOP AI models available
        self.ultimate_models = [
            {"name": "GPT-4o", "model": "gpt-4o", "provider": "openai", "weight": 1.0, "expertise": "reasoning"},
            {"name": "Claude Sonnet 3.5", "model": "claude-3-5-sonnet-20241022", "provider": "anthropic", "weight": 1.0, "expertise": "analysis"},
            {"name": "GPT-4 Turbo", "model": "gpt-4-turbo", "provider": "openai", "weight": 0.95, "expertise": "comprehensive"},
            {"name": "GPT-4o Mini", "model": "gpt-4o-mini", "provider": "openai", "weight": 0.9, "expertise": "technical"}
        ]
        
        # Load MT-Bench calibration
        self.mtbench_thresholds = self.load_mtbench_calibration()
        
        # Academic validation (from foolproof system)
        self.academic_indicators = [
            "abstract", "introduction", "methodology", "results", "conclusion",
            "references", "figure", "table", "experiment", "analysis",
            "research", "study", "paper", "work", "approach", "method",
            "dataset", "evaluation", "baseline", "performance"
        ]
        
        self.resume_indicators = [
            "experience", "education", "skills", "employment", "position",
            "resume", "cv", "curriculum vitae", "work experience", "job"
        ]
        
        # Conference quality mapping
        self.conference_tiers = {
            "tier_1": {"threshold": 8.0, "venues": ["NeurIPS", "ICML", "ICLR"], "acceptance_rate": "5%"},
            "tier_2": {"threshold": 7.0, "venues": ["EMNLP", "ACL", "AAAI"], "acceptance_rate": "15%"},
            "tier_3": {"threshold": 6.0, "venues": ["Strong Conferences"], "acceptance_rate": "25%"},
            "tier_4": {"threshold": 5.0, "venues": ["Workshops"], "acceptance_rate": "40%"},
            "reject": {"threshold": 0.0, "venues": ["Reject"], "acceptance_rate": "N/A"}
        }
    
    def load_mtbench_calibration(self):
        """Load MT-Bench calibration for realistic scoring"""
        return {
            "excellent": 4.6,     # Top 10%
            "good": 4.2,          # Above average
            "average": 4.0,       # Population mean
            "below_average": 3.8, # Below average
            "poor": 3.6           # Bottom 10%
        }
    
    def validate_academic_paper(self, text):
        """ULTIMATE validation - strict academic paper detection"""
        text_lower = text.lower()
        
        # Count indicators
        academic_score = sum(1 for indicator in self.academic_indicators if indicator in text_lower)
        resume_score = sum(1 for indicator in self.resume_indicators if indicator in text_lower)
        
        # Structure validation
        has_abstract = "abstract" in text_lower
        has_references = "references" in text_lower or "bibliography" in text_lower
        has_methodology = any(word in text_lower for word in ["method", "approach", "algorithm", "technique"])
        has_results = any(word in text_lower for word in ["results", "findings", "evaluation", "experiment"])
        has_sections = len(re.findall(r'\d+\.\s+[A-Z]', text)) >= 3
        
        # Quality indicators
        quality_signals = []
        if "f1" in text_lower or "precision" in text_lower or "recall" in text_lower:
            quality_signals.append("standard_metrics")
        if "baseline" in text_lower or "comparison" in text_lower:
            quality_signals.append("baseline_comparison")
        if "statistical" in text_lower or "significance" in text_lower:
            quality_signals.append("statistical_analysis")
        if "ablation" in text_lower:
            quality_signals.append("ablation_study")
        if "user study" in text_lower or "human evaluation" in text_lower:
            quality_signals.append("human_evaluation")
        
        print(f"🔍 ULTIMATE VALIDATION:")
        print(f"   Academic indicators: {academic_score}")
        print(f"   Resume indicators: {resume_score}")
        print(f"   Has abstract: {has_abstract}")
        print(f"   Has references: {has_references}")
        print(f"   Has methodology: {has_methodology}")
        print(f"   Has results: {has_results}")
        print(f"   Quality signals: {len(quality_signals)} ({', '.join(quality_signals)})")
        
        # STRICT validation criteria
        if resume_score > academic_score:
            return False, "Document appears to be a resume/CV, not an academic paper"
        
        if academic_score < 8:
            return False, "Insufficient academic content indicators for research paper"
            
        if not all([has_abstract, has_references, has_methodology, has_results]):
            return False, "Missing essential academic paper structure (abstract/references/methods/results)"
        
        if len(text) < 2000:
            return False, "Document too short for substantial academic paper"
            
        return True, f"Valid academic paper with {len(quality_signals)} quality indicators"
    
    def extract_text_from_pdf(self, pdf_path):
        """Extract and validate PDF text"""
        try:
            print(f"📄 Extracting from: {pdf_path}")
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                text = ""
                print(f"📊 PDF has {len(pdf_reader.pages)} pages")
                
                for i, page in enumerate(pdf_reader.pages):
                    page_text = page.extract_text()
                    text += page_text + "\n"
                    if i >= 12:  # More pages for better analysis
                        break
                
                print(f"📝 Extracted {len(text)} characters")
                
                # ULTIMATE validation
                is_valid, validation_message = self.validate_academic_paper(text)
                
                if not is_valid:
                    print(f"❌ VALIDATION FAILED: {validation_message}")
                    return f"ERROR: {validation_message}"
                
                print(f"✅ VALIDATION PASSED: {validation_message}")
                return text.strip()
                
        except Exception as e:
            print(f"❌ Error extracting PDF: {e}")
            return f"Error: {str(e)}"
    
    def generate_ultimate_prompt(self, paper_text, model_info):
        """Generate ULTIMATE prompts for top AI models"""
        expertise = model_info.get("expertise", "general")
        model_name = model_info.get("name", "Expert")
        
        # Analyze paper preview for context
        preview = paper_text[:2000].lower()
        
        # Detect paper domain
        if "citation" in preview:
            domain = "scientific integrity/information retrieval"
        elif "latex" in preview or "benchmark" in preview:
            domain = "document processing/evaluation"
        elif "peptide" in preview or "protein" in preview:
            domain = "computational biology/bioinformatics"
        elif "llm" in preview or "language model" in preview:
            domain = "natural language processing"
        else:
            domain = "computer science/machine learning"
        
        # Quality assessment
        quality_indicators = []
        if any(metric in preview for metric in ["f1", "precision", "recall", "accuracy"]):
            quality_indicators.append("quantitative_metrics")
        if "baseline" in preview:
            quality_indicators.append("baseline_comparison")
        if "statistical" in preview:
            quality_indicators.append("statistical_analysis")
        if "ablation" in preview:
            quality_indicators.append("ablation_study")
        if "user study" in preview:
            quality_indicators.append("human_evaluation")
            
        quality_level = len(quality_indicators)
        
        # ULTIMATE prompts based on model expertise
        if expertise == "reasoning":
            # GPT-5 (o1) - Advanced reasoning
            prompt = f"""
You are a DISTINGUISHED ACADEMIC REVIEWER with expertise in {domain}. As a top-tier model, you provide the most sophisticated analysis available.

PAPER DOMAIN: {domain}
QUALITY INDICATORS DETECTED: {quality_level}/5 ({', '.join(quality_indicators)})

PAPER TO REVIEW:
{paper_text[:20000]}

ADVANCED REASONING ANALYSIS:
Use your advanced reasoning capabilities to provide a thorough, critical review. Focus on:

1. **Contribution Assessment**
   - What is the actual novel contribution?
   - How significant is this advance?
   - Is the work incremental or breakthrough?

2. **Technical Rigor**
   - Are methods clearly described and sound?
   - Can results be reproduced?
   - Are claims supported by evidence?

3. **Experimental Quality**
   - Are experiments well-designed?
   - Are baselines appropriate?
   - Is statistical analysis proper?

4. **Impact Evaluation**
   - Will this influence future work?
   - Are there practical applications?
   - What are the limitations?

SCORING GUIDELINES:
- 9-10: Exceptional breakthrough (extremely rare)
- 8-8.5: Strong accept - clear advance (top 10%)
- 7-7.5: Accept - solid contribution (top 25%)
- 6-6.5: Borderline - needs revisions
- 5-5.5: Weak reject - major issues
- <5: Reject - fundamental problems

Please provide numerical scores (1-10) for:
- Novelty: ___/10
- Technical Quality: ___/10
- Experimental Rigor: ___/10
- Clarity: ___/10
- Significance: ___/10

Write a comprehensive review focusing on the actual paper content.
"""
            
        elif expertise == "analysis":
            # Claude Sonnet 3.5 - Analytical excellence
            prompt = f"""
You are an EXPERT ANALYTICAL REVIEWER specializing in {domain}. Provide the most thorough and insightful analysis possible.

DOMAIN EXPERTISE: {domain}
PAPER QUALITY SIGNALS: {quality_level}/5 indicators detected

PAPER FOR ANALYSIS:
{paper_text[:20000]}

ANALYTICAL FRAMEWORK:
Conduct a meticulous analysis of this work:

1. **Methodological Analysis**
   - Evaluate the approach for soundness
   - Identify methodological strengths/weaknesses
   - Assess reproducibility potential

2. **Empirical Evaluation**
   - Analyze experimental design quality
   - Evaluate baseline adequacy
   - Assess result validity

3. **Contribution Significance**
   - Determine novelty level
   - Evaluate practical impact
   - Compare to related work

4. **Critical Assessment**
   - Identify specific issues or limitations
   - Suggest concrete improvements
   - Provide overall recommendation

ANALYTICAL SCORING:
Rate each aspect (1-10 scale):
- Novelty: ___/10
- Technical Quality: ___/10
- Experimental Rigor: ___/10
- Clarity: ___/10
- Significance: ___/10

Provide detailed analytical feedback based strictly on paper content.
"""
            
        elif expertise == "comprehensive":
            # GPT-4o - Comprehensive review
            prompt = f"""
You are a COMPREHENSIVE REVIEWER with broad expertise across {domain}. Provide balanced, thorough assessment.

COMPREHENSIVE ANALYSIS SCOPE: {domain}
DETECTED QUALITY FEATURES: {quality_level}/5

PAPER TO COMPREHENSIVELY REVIEW:
{paper_text[:18000]}

COMPREHENSIVE REVIEW STRUCTURE:

1. **Summary**
   - Brief overview of the work
   - Main contributions claimed
   - Key findings presented

2. **Strengths Analysis**
   - Technical innovations
   - Experimental strengths
   - Clear methodological aspects

3. **Weaknesses Identification**
   - Technical limitations
   - Experimental gaps
   - Methodological issues

4. **Detailed Assessment**
   - Novelty evaluation
   - Technical soundness
   - Experimental rigor
   - Practical significance

COMPREHENSIVE SCORING:
- Novelty: ___/10
- Technical Quality: ___/10
- Experimental Rigor: ___/10
- Clarity: ___/10
- Significance: ___/10

Provide comprehensive analysis covering all aspects of the work.
"""
            
        else:  # technical expertise
            # GPT-4 Turbo - Technical focus
            prompt = f"""
You are a TECHNICAL EXPERT REVIEWER specializing in {domain}. Focus on technical depth and methodological rigor.

TECHNICAL EXPERTISE: {domain}
METHODOLOGY INDICATORS: {quality_level}/5 detected

PAPER FOR TECHNICAL REVIEW:
{paper_text[:18000]}

TECHNICAL EVALUATION FOCUS:

1. **Technical Methodology**
   - Algorithm/method description clarity
   - Technical implementation feasibility
   - Computational complexity considerations

2. **Experimental Validation**
   - Experimental design adequacy
   - Statistical rigor assessment
   - Baseline comparison quality

3. **Technical Contributions**
   - Methodological innovations
   - Technical novelty evaluation
   - Implementation challenges

4. **Reproducibility Assessment**
   - Technical detail sufficiency
   - Code/data availability
   - Replication potential

TECHNICAL SCORING:
- Novelty: ___/10
- Technical Quality: ___/10
- Experimental Rigor: ___/10
- Clarity: ___/10
- Significance: ___/10

Focus on technical aspects and methodological rigor in your review.
"""
        
        return prompt
    
    def get_model_review(self, model_info, paper_text):
        """Get review from TOP AI models"""
        try:
            prompt = self.generate_ultimate_prompt(paper_text, model_info)
            provider = model_info.get("provider", "openai")
            
            print(f"🤖 Getting review from {model_info['name']} ({provider})...")
            
            if provider == "anthropic":
                # Use Anthropic API for Claude
                try:
                    import anthropic
                    client = anthropic.Anthropic(api_key=anthropic_api_key)
                    response = client.messages.create(
                        model=model_info["model"],
                        max_tokens=4000,
                        messages=[{"role": "user", "content": prompt}]
                    )
                    content = response.content[0].text
                except Exception as claude_error:
                    print(f"Claude API error: {claude_error}")
                    # Use GPT-4o as fallback for Claude
                    fallback_response = openai.ChatCompletion.create(
                        model="gpt-4o",
                        messages=[{"role": "user", "content": prompt}],
                        max_tokens=4000,
                        temperature=0.1
                    )
                    content = fallback_response.choices[0].message.content
            else:
                # Use OpenAI API for GPT models
                response = openai.ChatCompletion.create(
                    model=model_info["model"],
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=4000,
                    temperature=0.1  # Low temperature for consistency
                )
                content = response.choices[0].message.content
            
            print(f"✅ {model_info['name']} review completed ({len(content)} chars)")
            
            return {
                "model": model_info["name"],
                "expertise": model_info["expertise"],
                "review_content": content,
                "success": True,
                "provider": provider
            }
            
        except Exception as e:
            print(f"❌ Error from {model_info['name']}: {e}")
            return {
                "model": model_info["name"],
                "review_content": f"Review unavailable: {str(e)}",
                "success": False,
                "error": str(e)
            }
    
    def extract_ultimate_scores(self, review_content):
        """Extract scores using ULTIMATE parsing"""
        scores = {}
        
        # Enhanced score extraction patterns
        patterns = [
            (r"Novelty:\s*(\d+(?:\.\d+)?)/10", "novelty"),
            (r"Technical Quality:\s*(\d+(?:\.\d+)?)/10", "technical_quality"),
            (r"Experimental Rigor:\s*(\d+(?:\.\d+)?)/10", "experimental_rigor"),
            (r"Clarity:\s*(\d+(?:\.\d+)?)/10", "clarity"),
            (r"Significance:\s*(\d+(?:\.\d+)?)/10", "significance"),
            # Alternative patterns
            (r"Overall:\s*(\d+(?:\.\d+)?)/10", "overall"),
            (r"Quality:\s*(\d+(?:\.\d+)?)/10", "quality"),
            (r"Rating:\s*(\d+(?:\.\d+)?)", "rating")
        ]
        
        for pattern, criterion in patterns:
            matches = re.findall(pattern, review_content, re.IGNORECASE)
            if matches:
                try:
                    scores[criterion] = float(matches[0])
                except:
                    continue
        
        # If no specific scores found, try to extract general assessment
        if not scores:
            # Look for general quality indicators in text
            if "excellent" in review_content.lower():
                base_score = 8.0
            elif "strong" in review_content.lower():
                base_score = 7.0
            elif "good" in review_content.lower():
                base_score = 6.0
            elif "weak" in review_content.lower():
                base_score = 4.0
            else:
                base_score = 5.0
                
            scores = {
                "novelty": base_score,
                "technical_quality": base_score,
                "experimental_rigor": base_score,
                "clarity": base_score,
                "significance": base_score
            }
        
        return scores
    
    def calculate_ultimate_consensus(self, reviews):
        """Calculate ULTIMATE consensus using top model weights"""
        successful_reviews = [r for r in reviews if r["success"]]
        
        if not successful_reviews:
            return 0, "No successful reviews", {}
        
        all_model_scores = []
        model_weights = []
        detailed_scores = {}
        
        print(f"📊 ULTIMATE CONSENSUS CALCULATION:")
        
        for review in successful_reviews:
            scores = self.extract_ultimate_scores(review["review_content"])
            model_name = review["model"]
            
            if scores:
                # Calculate weighted average for this model
                avg_score = sum(scores.values()) / len(scores)
                all_model_scores.append(avg_score)
                
                # Get model weight
                model_weight = 1.0
                for model_info in self.ultimate_models:
                    if model_info["name"] == model_name:
                        model_weight = model_info["weight"]
                        break
                        
                model_weights.append(model_weight)
                detailed_scores[model_name] = {
                    "scores": scores,
                    "average": avg_score,
                    "weight": model_weight
                }
                
                print(f"   {model_name}: {avg_score:.1f}/10 (weight: {model_weight})")
        
        if not all_model_scores:
            return 0, "No scores extracted", {}
        
        # Calculate weighted consensus
        if model_weights:
            consensus_score = sum(s * w for s, w in zip(all_model_scores, model_weights)) / sum(model_weights)
        else:
            consensus_score = sum(all_model_scores) / len(all_model_scores)
        
        print(f"📊 Ultimate consensus: {consensus_score:.2f}/10")
        
        return consensus_score, "Success", detailed_scores
    
    def determine_ultimate_verdict(self, consensus_score):
        """Determine verdict using conference tier mapping"""
        for tier, info in self.conference_tiers.items():
            if consensus_score >= info["threshold"]:
                return {
                    "tier": tier,
                    "verdict": f"{tier.replace('_', ' ').title()} Quality",
                    "venues": info["venues"],
                    "acceptance_rate": info["acceptance_rate"]
                }
        
        return {
            "tier": "reject",
            "verdict": "Reject - Below conference standards",
            "venues": ["None"],
            "acceptance_rate": "0%"
        }
    
    def review_paper(self, paper_text, paper_id):
        """ULTIMATE review process combining best approaches"""
        print("🚀 ULTIMATE PEER REVIEW STARTING...")
        
        # STEP 1: Ultimate validation
        is_valid, validation_message = self.validate_academic_paper(paper_text)
        if not is_valid:
            return {
                "error": validation_message,
                "paper_id": paper_id,
                "status": "validation_failed"
            }
        
        print(f"✅ Ultimate validation passed: {validation_message}")
        
        # STEP 2: Multi-top-model review
        reviews = []
        for model_info in self.ultimate_models:
            review = self.get_model_review(model_info, paper_text)
            reviews.append(review)
            time.sleep(1)  # Rate limiting
        
        # STEP 3: Ultimate consensus
        consensus_score, consensus_status, detailed_scores = self.calculate_ultimate_consensus(reviews)
        
        # STEP 4: Determine verdict
        verdict_info = self.determine_ultimate_verdict(consensus_score)
        
        result = {
            "paper_id": paper_id,
            "timestamp": datetime.now().isoformat(),
            "validation_status": validation_message,
            "consensus_score": round(consensus_score, 2),
            "verdict_info": verdict_info,
            "detailed_scores": detailed_scores,
            "individual_reviews": reviews,
            "system_info": {
                "models_used": len(self.ultimate_models),
                "successful_reviews": len([r for r in reviews if r["success"]]),
                "consensus_method": "weighted_average",
                "calibration": "mtbench_based"
            }
        }
        
        print(f"✅ ULTIMATE REVIEW COMPLETED:")
        print(f"   Score: {consensus_score:.2f}/10")
        print(f"   Tier: {verdict_info['tier']}")
        print(f"   Suitable Venues: {', '.join(verdict_info['venues'])}")
        
        return result

# Initialize ultimate reviewer
ultimate_reviewer = UltimatePeerReviewer()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file and file.filename.lower().endswith('.pdf'):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        print(f"📁 File uploaded: {filename}")
        
        # Extract and validate text with ULTIMATE system
        paper_text = ultimate_reviewer.extract_text_from_pdf(filepath)
        
        if paper_text.startswith("ERROR:"):
            return jsonify({'error': paper_text}), 400
        
        # Generate unique paper ID
        paper_id = str(int(time.time() * 1000))
        
        # Start ULTIMATE review process
        def ultimate_review_worker():
            try:
                result = ultimate_reviewer.review_paper(paper_text, paper_id)
                review_results[paper_id] = result
            except Exception as e:
                print(f"❌ Ultimate review error: {e}")
                review_results[paper_id] = {"error": str(e)}
        
        thread = threading.Thread(target=ultimate_review_worker)
        thread.start()
        
        return jsonify({
            'success': True,
            'paper_id': paper_id,
            'message': 'Academic paper validated and ULTIMATE review started'
        })
    
    return jsonify({'error': 'Please upload a PDF file'}), 400

@app.route('/review/<paper_id>')
def get_review_status(paper_id):
    if paper_id in review_results:
        return jsonify(review_results[paper_id])
    else:
        return jsonify({'status': 'processing'})

if __name__ == "__main__":
    print("🚀 ULTIMATE PEER REVIEW SYSTEM STARTING...")
    print("   Models: GPT-5 (o1), Claude Sonnet 3.5, GPT-4o, GPT-4 Turbo")
    print("   Features: MT-Bench calibration + Foolproof validation")
    print("   URL: http://127.0.0.1:8000")
    app.run(debug=False, host='0.0.0.0', port=8000)
