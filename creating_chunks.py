import whisper
import json
import os

model = whisper.load_model("large-v2", device="cpu")
audios = os.listdir('audios')
for audio in audios:
    if ('_' in audio):
        number = audio.split('_')[0]
        title = audio.split('_')[1]

        result = model.transcribe(
            audio=f'audios/{audio}',
            language='hi',
            task='translate',
            fp16=False,
            word_timestamps=False
        )
        chunks = []
        for segment in result['segments']:
            chunks.append(
                {'Number': number, 'title': title, 'start': segment['start'], 'end': segment['end'], 'text': segment['text']})
        chunk_with_metadata = {"chunks": chunks, 'text': result['text']}

        with open(f'jsons/{audio}.json', 'w') as f:
            json.dump(chunk_with_metadata, f)
