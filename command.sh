# commands for installing the prerequisite of IndicTranslator models
git clone https://github.com/AI4Bharat/IndicTrans2.git
cd /IndicTrans2/huggingface_interface
capture
python3 -m pip install nltk sacremoses pandas regex mock transformers>=4.33.2 mosestokenizer
python3 -c "import nltk; nltk.download('punkt')"
python3 -m pip install bitsandbytes scipy accelerate datasets
python3 -m pip install sentencepiece
python3 -m pip install indic-nlp-library


git clone https://github.com/VarunGumma/IndicTransToolkit.git
cd IndicTransToolkit
python3 -m pip install --editable ./
cd ..

#commands for installing the NLTM-LID/LID-version-2.0 for audio language identification
pip install pip==24.0 # for downgrade the pip
pip install fairseq==0.12.2
git clone https://github.com/NLTM-LID/LID-version-2.0.git
cd LID-version-2.0
pip install -r requirement.txt
cd ..
#prerequisites for tts model api from huggingface
curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.python.sh | bash
sudo apt-get install git-lfs
git lfs install
git clone https://huggingface.co/k-m-irfan/Fastspeech2_HS_Flask_API
cd Fastspeech2_HS_Flask_API
pip install -r requirements.txt
pip install scipy==1.9.1
# to run the streamlit client app
streamlit run --server.port 8500 clients/main.py