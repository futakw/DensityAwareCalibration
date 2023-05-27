python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
conda install faiss-gpu cudatoolkit=11.0 -c pytorch