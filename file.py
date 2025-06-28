import os
import json
import faiss
import numpy as np
from sklearn.decomposition import TruncatedSVD
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from flask import Flask, request, jsonify, send_from_directory

app = Flask(__name__, static_folder='.')

question_encoder = None
context_encoder = None
global_trained_models = {}
faiss_indices = {'tfidf': None, 'lsa': None, 'dpr': None}
faiss_data = []

ESTABLISHED_FIELDS = [
    'artificial intelligence', 
    'machine learning',
    'deep learning',
    'computer vision',
    'reinforcement learning'
]

def load_dpr_models():
    """Lazy load DPR models only when needed"""
    global question_encoder, context_encoder
    if question_encoder is None or context_encoder is None:
        try:
            print("Loading DPR models...")
            question_encoder = SentenceTransformer('facebook-dpr-question_encoder-single-nq-base')
            context_encoder = SentenceTransformer('facebook-dpr-ctx_encoder-single-nq-base')
            print("DPR models loaded successfully")
        except Exception as e:
            print(f"Warning: Could not load DPR models: {e}")
            return False
    return True

def generate_author_context(author):
    """Generate context string for author"""
    name = author.get('display_name') or author.get('name', '')
    concept = author.get('source_concept', '')
    affiliation = author.get('affiliation', '') or author.get('institution', '')
    works = author.get('works_count', 0) or author.get('paper_count', 0) or author.get('total_papers', 0)
    citations = author.get('cited_by_count', 0) or author.get('total_citations', 0)
    return f"Author: {name}. Research field: {concept}. Affiliation: {affiliation}. Works: {works}. Citations: {citations}. Keywords: {concept}, artificial intelligence, machine learning, research, {name}."

def load_json_file(file_path):
    """Load JSON file with proper encoding handling"""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f'File not found: {file_path}')
    for encoding in ['utf-8', 'latin-1', 'cp1252']:
        try:
            with open(file_path, 'r', encoding=encoding) as f:
                return json.load(f)
        except (UnicodeDecodeError, json.JSONDecodeError):
            continue
    raise Exception(f'Could not decode file: {file_path}')

def load_all_data_for_training():
    """Load ALL data (authors, papers, institutions) from all concepts"""
    try:
        print("Loading ALL data from all concepts for training...")
        all_data = []
        
        author_files = [
            'top_ai_authors_with_papers.json',
            'top_cv_authors.json',
            'top_dl_authors_with_papers.json',
            'top_ml_authors.json',
            'top_rl_authors_with_papers.json'
        ]
        concept_mapping = {
            'top_ai_authors_with_papers.json': 'artificial_intelligence',
            'top_cv_authors.json': 'computer_vision',
            'top_dl_authors_with_papers.json': 'deep_learning',
            'top_ml_authors.json': 'machine_learning',
            'top_rl_authors_with_papers.json': 'reinforcement_learning'
        }
        
        for filename in author_files:
            try:
                file_path = os.path.join('searching_codes', 'top_authors_concept', filename)
                if os.path.exists(file_path):
                    authors_data = load_json_file(file_path)
                    concept = concept_mapping[filename]
                    if isinstance(authors_data, dict) and "authors" in authors_data:
                        authors_data = authors_data["authors"]
                    if isinstance(authors_data, list):
                        for author in authors_data:
                            if isinstance(author, dict):
                                author['source_concept'] = concept
                                author['data_type'] = 'author'
                                all_data.append(author)
            except Exception as e:
                print(f"Error loading authors from {filename}: {e}")
        
        paper_files = [
            'artificial_intelligence.json',
            'computer_vision.json',
            'deep_learning.json',
            'machine_learning.json',
            'reinforcement_learning.json'
        ]
        for filename in paper_files:
            try:
                file_path = os.path.join('searching_codes', 'papers_by_concept', filename)
                if os.path.exists(file_path):
                    papers_data = load_json_file(file_path)
                    concept = filename.replace('.json', '')
                    if isinstance(papers_data, list):
                        for paper in papers_data:
                            if isinstance(paper, dict):
                                paper['source_concept'] = concept
                                paper['data_type'] = 'paper'
                                all_data.append(paper)
            except Exception as e:
                print(f"Error loading papers from {filename}: {e}")
        
        institution_files = [
            'artificial_intelligence.json',
            'computer_vision.json',
            'deep_learning.json',
            'machine_learning.json',
            'reinforcement_learning.json'
        ]
        for filename in institution_files:
            try:
                file_path = os.path.join('searching_codes', 'institutions_by_domain', filename)
                if os.path.exists(file_path):
                    institutions_data = load_json_file(file_path)
                    concept = filename.replace('.json', '')
                    if isinstance(institutions_data, list):
                        for inst in institutions_data:
                            if isinstance(inst, list) and len(inst) >= 2:
                                institution = {
                                    'name': str(inst[0]),
                                    'score': inst[1] if len(inst) > 1 else 0,
                                    'source_concept': concept,
                                    'data_type': 'institution'
                                }
                                all_data.append(institution)
                            elif isinstance(inst, dict):
                                inst['source_concept'] = concept
                                inst['data_type'] = 'institution'
                                all_data.append(inst)
            except Exception as e:
                print(f"Error loading institutions from {filename}: {e}")
        
        print(f"Total data loaded: {len(all_data)} items")
        return all_data
    except Exception as e:
        print(f"Error loading all data: {e}")
        return []

def train_global_models():
    """Train all models on ALL available data with FAISS indexing"""
    global global_trained_models, faiss_indices, faiss_data
    try:
        print("Training global models with FAISS indexing on ALL available data...")
        all_data = load_all_data_for_training()
        if not all_data:
            print("No data available for training")
            return False
        
        faiss_data = all_data
        contexts = []
        metadata = []
        
        for item in all_data:
            context = ""
            if item.get('data_type') == 'author':
                context = generate_author_context(item)
            elif item.get('data_type') == 'paper':
                title = item.get('title', '')
                abstract = item.get('abstract', '') or item.get('summary', '')
                year = item.get('year', '') or item.get('publication_year', '')
                citations = item.get('cited_by_count', 0) or item.get('citations', 0)
                concepts = ", ".join(item.get('concepts', [])) or item.get('source_concept', '')
                context = f"Paper: {title}. Abstract: {abstract}. Year: {year}. Citations: {citations}. Concepts: {concepts}."
            elif item.get('data_type') == 'institution':
                name = item.get('name', '') or item.get('display_name', '')
                country = item.get('country', '') or item.get('country_code', '')
                score = item.get('score', 0) or item.get('institution_score', 0)
                concepts = ", ".join(item.get('concepts', [])) or item.get('source_concept', '')
                context = f"Institution: {name}. Country: {country}. Score: {score}. Concepts: {concepts}."
            
            if context:
                contexts.append(context)
                metadata.append({
                    'type': item.get('data_type', 'unknown'),
                    'item': item
                })
        
        if not contexts:
            print("No valid contexts created for training")
            return False
        
        print(f"Training on {len(contexts)} contexts...")
        
        print("Training TF-IDF + FAISS...")
        tfidf_vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            ngram_range=(1, 3),
            min_df=2,
            max_df=0.9,
            sublinear_tf=True
        )
        tfidf_matrix = tfidf_vectorizer.fit_transform(contexts)
        tfidf_dense = tfidf_matrix.toarray().astype('float32')
        faiss.normalize_L2(tfidf_dense)
        tfidf_index = faiss.IndexFlatIP(tfidf_dense.shape[1])
        tfidf_index.add(tfidf_dense)
        faiss_indices['tfidf'] = tfidf_index
        
        print("Training LSA + FAISS...")
        lsa_model = TruncatedSVD(n_components=100, random_state=42)
        lsa_matrix = lsa_model.fit_transform(tfidf_matrix)
        lsa_dense = lsa_matrix.astype('float32')
        faiss.normalize_L2(lsa_dense)
        lsa_index = faiss.IndexFlatIP(lsa_dense.shape[1])
        lsa_index.add(lsa_dense)
        faiss_indices['lsa'] = lsa_index
        
        print("Training DPR + FAISS...")
        try:
            dpr_model = SentenceTransformer('all-MiniLM-L6-v2')
            dpr_embeddings = dpr_model.encode(contexts, convert_to_tensor=False, show_progress_bar=False)
            dpr_dense = dpr_embeddings.astype('float32')
            faiss.normalize_L2(dpr_dense)
            dpr_index = faiss.IndexFlatIP(dpr_dense.shape[1])
            dpr_index.add(dpr_dense)
            faiss_indices['dpr'] = dpr_index
            dpr_success = True
        except Exception as e:
            print(f"DPR training failed: {e}")
            dpr_success = False
        
        global_trained_models = {
            'contexts': contexts,
            'metadata': metadata,
            'tfidf_vectorizer': tfidf_vectorizer,
            'lsa_model': lsa_model,
            'dpr_model': dpr_model if dpr_success else None,
            'dpr_available': dpr_success,
            'trained_on_all_data': True,
            'total_items': len(all_data)
        }
        
        print("Global training completed successfully!")
        return True
    except Exception as e:
        print(f"Global training error: {e}")
        return False

@app.route('/')
def index():
    return send_from_directory('.', 'index.html')

@app.route('/train_global_models', methods=['POST'])
def train_global_models_endpoint():
    success = train_global_models()
    if success:
        return jsonify({
            'success': True,
            'message': 'Global models trained successfully with FAISS indexing'
        })
    else:
        return jsonify({'error': 'Failed to train global models'}), 500

@app.route('/search_by_topic', methods=['POST'])
def search_by_topic():
    global global_trained_models, faiss_indices, faiss_data
    try:
        data = request.get_json()
        topic = data.get('topic', '').strip()
        if not topic:
            return jsonify({'error': 'Topic is required'}), 400
        
        if not global_trained_models:
            return jsonify({'error': 'Global models not trained. Please train models first.'}), 400
        
        enhanced_query = f"{topic} research artificial intelligence machine learning"
        results = {'success': True, 'topic': topic}
        
        def search_method(method_name, query_vector, index, top_k=5):
            try:
                if isinstance(query_vector, np.ndarray):
                    query_dense = query_vector.reshape(1, -1).astype('float32')
                else:
                    query_dense = query_vector.toarray().astype('float32')
                
                norm = np.linalg.norm(query_dense)
                if norm > 0:
                    query_dense /= norm
                else:
                    return {"error": "Zero query vector"}
                
                similarities, indices = index.search(query_dense, top_k)
                method_results = []
                
                for sim, idx in zip(similarities[0], indices[0]):
                    if idx < len(faiss_data):
                        item = faiss_data[idx]
                        result_item = {
                            'type': item.get('data_type', 'unknown'),
                            'similarity': float(sim * 100)
                        }
                        
                        if result_item['type'] == 'author':
                            result_item.update({
                                'name': item.get('display_name') or item.get('name', ''),
                                'affiliation': item.get('affiliation', '') or item.get('institution', ''),
                                'concept': item.get('source_concept', ''),
                                'works_count': item.get('works_count', 0) or item.get('paper_count', 0),
                                'cited_by_count': item.get('cited_by_count', 0) or item.get('total_citations', 0)
                            })
                        elif result_item['type'] == 'paper':
                            result_item.update({
                                'title': item.get('title', ''),
                                'year': item.get('year', '') or item.get('publication_year', ''),
                                'citations': item.get('cited_by_count', 0) or item.get('citations', 0),
                                'concepts': item.get('concepts', []) or [item.get('source_concept', '')]
                            })
                        elif result_item['type'] == 'institution':
                            result_item.update({
                                'name': item.get('name', '') or item.get('display_name', ''),
                                'country': item.get('country', '') or item.get('country_code', ''),
                                'score': item.get('score', 0) or item.get('institution_score', 0),
                                'concepts': [item.get('source_concept', '')]
                            })
                        
                        method_results.append(result_item)
                
                return method_results
            except Exception as e:
                return {"error": str(e)}
        
        if faiss_indices['tfidf'] is not None:
            query_tfidf = global_trained_models['tfidf_vectorizer'].transform([enhanced_query])
            results['tfidf_results'] = search_method('tfidf', query_tfidf, faiss_indices['tfidf'])
        
        if faiss_indices['lsa'] is not None:
            query_tfidf = global_trained_models['tfidf_vectorizer'].transform([enhanced_query])
            query_lsa = global_trained_models['lsa_model'].transform(query_tfidf)
            results['lsa_results'] = search_method('lsa', query_lsa, faiss_indices['lsa'])
        
        if faiss_indices['dpr'] is not None and global_trained_models['dpr_available']:
            query_embedding = global_trained_models['dpr_model'].encode([enhanced_query], convert_to_tensor=False)
            results['dpr_results'] = search_method('dpr', query_embedding, faiss_indices['dpr'])
        
        results['total_items_searched'] = len(faiss_data)
        return jsonify(results)
    except Exception as e:
        return jsonify({'error': f'Search error: {str(e)}'}), 500

@app.route('/analyze_institutions', methods=['POST'])
def analyze_institutions():
    global global_trained_models
    try:
        data = request.get_json()
        topic = data.get('topic', '').strip()
        limit = data.get('limit', 10)
        if not topic:
            return jsonify({'error': 'Topic is required'}), 400
        
        if not global_trained_models or not global_trained_models.get('dpr_available'):
            return jsonify({'error': 'DPR model not available. Train global models first.'}), 400
        
        all_data = load_all_data_for_training()
        if not all_data:
            return jsonify({'error': 'No data available'}), 404
        
        all_authors = [item for item in all_data if item.get('data_type') == 'author']
        if not all_authors:
            return jsonify({'error': 'No author data available'}), 404
        
        dpr_model = global_trained_models['dpr_model']
        topic_embedding = dpr_model.encode([topic], convert_to_tensor=False)[0]
        topic_embedding_norm = topic_embedding / np.linalg.norm(topic_embedding)
        
        institution_stats = {}
        all_concepts = set()
        max_papers = 0
        
        for author in all_authors:
            concept = author.get('source_concept', '')
            if concept:
                all_concepts.add(concept)
            
            institution_name = (author.get('institution') or 
                              author.get('affiliation') or 
                              author.get('last_known_institution') or 
                              (author.get('affiliations') and author['affiliations'][0] and 
                               author['affiliations'][0].get('institution') and 
                               author['affiliations'][0]['institution'].get('display_name')) or 
                              'Unknown Institution').strip()
            
            if institution_name == 'Unknown Institution':
                continue
            
            if institution_name not in institution_stats:
                institution_stats[institution_name] = {
                    'name': institution_name,
                    'authors': [],
                    'total_papers': 0,
                    'total_citations': 0,
                    'author_contexts': []
                }
            
            institution_stats[institution_name]['authors'].append(author)
            paper_count = (author.get('works_count') or 
                          author.get('paper_count') or 
                          author.get('total_papers') or 0)
            citation_count = (author.get('cited_by_count') or 
                             author.get('total_citations') or 
                             author.get('citations') or 0)
            
            institution_stats[institution_name]['total_papers'] += paper_count
            institution_stats[institution_name]['total_citations'] += citation_count
            institution_stats[institution_name]['author_contexts'].append(generate_author_context(author))
            
            if institution_stats[institution_name]['total_papers'] > max_papers:
                max_papers = institution_stats[institution_name]['total_papers']
        
        institutions = []
        for inst_name, inst_data in institution_stats.items():

            if not inst_data['author_contexts']:
                continue
                
            author_embeddings = dpr_model.encode(inst_data['author_contexts'], convert_to_tensor=False)
            author_embeddings_norm = author_embeddings / np.linalg.norm(author_embeddings, axis=1, keepdims=True)
            
            similarities = np.dot(author_embeddings_norm, topic_embedding_norm)
            similarities = np.maximum(similarities, 0)  # Clip negative values
            s_sim = float(np.mean(similarities))
            
            s_impact = min(inst_data['total_citations'] / 10000.0, 1.0)
            s_prod = inst_data['total_papers'] / max_papers if max_papers > 0 else 0
            
            t_value = 0.5 if topic.lower() in [f.lower() for f in ESTABLISHED_FIELDS] else 0.2
            
            score = (s_sim * t_value) + s_impact + s_prod
            
            institutions.append({
                'name': inst_name,
                'papers': inst_data['total_papers'],
                'citations': inst_data['total_citations'],
                'authors': len(inst_data['authors']),
                'concepts': list(set(a.get('source_concept', '') for a in inst_data['authors'])),
                's_sim': s_sim,
                's_impact': s_impact,
                's_prod': s_prod,
                't_value': t_value,
                'score': float(score)
            })
        
        institutions.sort(key=lambda x: x['score'], reverse=True)
        for i, inst in enumerate(institutions):
            inst['rank'] = i + 1
        
        return jsonify({
            'success': True,
            'topic': topic,
            'institutions': institutions[:limit],
            'total_institutions': len(institutions),
            'total_authors_analyzed': len(all_authors),
            'concepts_analyzed': list(all_concepts),
            'max_papers': max_papers
        })
    except Exception as e:
        return jsonify({'error': f'Analysis error: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
