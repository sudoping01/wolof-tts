import os
from flask import Flask, request, send_file, jsonify
import torch
import numpy as np
import soundfile as sf
from io import BytesIO
import traceback


from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts
from removesilence import detect_silence, remove_silence  

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = '/tmp'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  


device = "cuda:0" if torch.cuda.is_available() else "cpu"

class TtsSynthesizer:
    def __init__(self):

        self.root_path       = "galsenai-xtts-wo-checkpoints/"
        self.checkpoint_path = os.path.join(self.root_path, "Anta_GPT_XTTS_Wo")
        self.model_path      = "best_model_89250.pth"
        self.xtts_config_path= os.path.join(self.checkpoint_path, "config.json")
        self.xtts_vocab      = os.path.join(self.root_path, "XTTS_v2.0_original_model_files", "vocab.json")
        
        self.model = None
        self.config = None
        self.load_model()
        
        self.reference_audio = os.path.join(self.root_path, "anta_sample.wav")
        self.gpt_cond_latent, self.speaker_embedding = self.get_conditioning_latents(self.reference_audio)

    def load_model(self):
        try:
            config = XttsConfig()
            config.load_json(self.xtts_config_path)
            self.config = config
            self.model = Xtts.init_from_config(config)
            self.model.load_checkpoint(
                config,
                checkpoint_path=os.path.join(self.checkpoint_path, self.model_path),
                vocab_path=self.xtts_vocab,
                use_deepspeed=False
            )
            self.model.to(device)
            print("Model loaded successfully!")
        except Exception as e:
            print("Error loading model:")
            traceback.print_exc()
            raise e

    def get_conditioning_latents(self, audio_path):
        try:
            return self.model.get_conditioning_latents(
                audio_path=[audio_path],
                gpt_cond_len=self.model.config.gpt_cond_len,
                max_ref_length=self.model.config.max_ref_len,
                sound_norm_refs=self.model.config.sound_norm_refs
            )
        except Exception as e:
            print("Error generating conditioning latents:")
            traceback.print_exc()
            raise e

    def synthesize(self, text, speed=1.06, language="wo", enable_text_splitting=True):
        try:
            result = self.model.inference(
                text=text.lower(),
                gpt_cond_latent=self.gpt_cond_latent,
                speaker_embedding=self.speaker_embedding,
                do_sample=False,
                speed=speed,
                language=language,
                enable_text_splitting=enable_text_splitting
            )
            audio_signal = result['wav']

            print("Audio shape:", audio_signal.shape)
            print("Audio min:", np.min(audio_signal), "max:", np.max(audio_signal))
            max_val = np.max(np.abs(audio_signal))
            if max_val == 0:
                print("Warning: Generated audio is all zeros!")
                return audio_signal

            audio_signal = audio_signal / max_val
            return audio_signal
        except Exception as e:
            print("Error during synthesis:")
            traceback.print_exc()
            raise e

    def remove_silence(self, input_audio_path, output_audio_path):
        try:
            silence_list = detect_silence(input_audio_path)
            remove_silence(input_audio_path, silence_list, output_audio_path)
            return output_audio_path
        except Exception as e:
            print("Error during silence removal:")
            traceback.print_exc()
            raise e

# Instantiate the synthesizer at startup.
tts_synthesizer = TtsSynthesizer()

@app.route('/predict', methods=['POST'])
def predict():
    """
    Expects a JSON payload like:
      { "text": "Your text to synthesize here" }
    Returns the synthesized audio as a WAV file.
    """
    data = request.get_json(force=True)
    if not data or 'text' not in data:
        return jsonify({"error": "No text provided"}), 400
    
    text = data['text']
    
    try:
        audio_signal = tts_synthesizer.synthesize(text)
        sample_rate = 24000
        
        buffer = BytesIO()
        sf.write(buffer, audio_signal, sample_rate, format='WAV')
        buffer.seek(0)

        return send_file(buffer, mimetype="audio/wav", as_attachment=True, download_name="generated_audio.wav")
    except Exception as e:
        print("Error in /predict endpoint:")
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        "status": "healthy",
        "device": device
    })

if __name__ == '__main__':

    port = int(os.environ.get('AIP_HTTP_PORT', 8080))
    app.run(host="0.0.0.0", port=port)
