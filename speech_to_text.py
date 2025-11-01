import whisper
import json
model = whisper.load_model("large-v2", device="cpu") 
result = model.transcribe(
    audio='audios/output_10s.mp3',
    language='hi',
    task='translate',
    fp16=False ,
    word_timestamps=False
)

chunks = []
for segment in result['segments']:
    chunks.append({'start': segment['start'] , 'end': segment['end'] , 'text': segment['text']})
print(chunks)

with open('output.json' , 'w') as f:
    json.dump(chunks , f)