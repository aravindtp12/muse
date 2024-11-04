from flask import Flask, render_template, request, jsonify
import os
from src.pdf_qa import create_vectorstore, create_qa_chain

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

qa_chain = None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    global qa_chain
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file and file.filename.endswith('.pdf'):
        filepath = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(filepath)
        
        # Create vector store and QA chain
        vectorstore = create_vectorstore(filepath)
        qa_chain = create_qa_chain(vectorstore)
        
        return jsonify({'message': 'PDF processed successfully'})
    
    return jsonify({'error': 'Invalid file type'}), 400

@app.route('/ask', methods=['POST'])
def ask_question():
    global qa_chain
    if qa_chain is None:
        return jsonify({'error': 'Please upload a PDF first'}), 400
    
    data = request.json
    question = data.get('question')
    if not question:
        return jsonify({'error': 'No question provided'}), 400
    
    response = qa_chain({"query": question})
    return jsonify({
        'answer': response['result'],
        'sources': [doc.metadata for doc in response['source_documents']]
    })

if __name__ == '__main__':
    app.run(debug=True) 