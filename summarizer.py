import assemblyai as aai
from transformers import BartForConditionalGeneration, BartTokenizer

# Set up AssemblyAI API key
aai.settings.api_key = "16b84bd8b8e14bf3acbc3717d0609648"

FILE_URL = "audio2.mp3"


model = BartForConditionalGeneration.from_pretrained('facebook/bart-large-cnn')
tokenizer = BartTokenizer.from_pretrained('facebook/bart-large-cnn')

def summarize_text(text, maxSummarylength=900):
    inputs = tokenizer.encode("summarize: " + text, return_tensors="pt", max_length=1024, truncation=True)
    summary_ids = model.generate(inputs, max_length=maxSummarylength, min_length=maxSummarylength // 2,
                                 length_penalty=2.0, num_beams=4, early_stopping=True)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

# Transcribe audio file and summarize the text
transcriber = aai.Transcriber()
transcript = transcriber.transcribe(FILE_URL,config=aai.TranscriptionConfig(summarization=True,summary_model=aai.SummarizationModel.informative,summary_type=aai.SummarizationType.bullets))
if transcript.status == aai.TranscriptStatus.error:
    print(transcript.error)
else:
    bullet_points=transcript.summary

if transcript.status == aai.TranscriptStatus.completed:
    transcribed_text = transcript.text
    #print("Transcribed Text:\n", transcribed_text)
    
    summarized_text = summarize_text(transcribed_text)
    #print("\nSummarized Text:\n", summarized_text)
    #print(len(summarized_text))
    
    # Save transcribed and summarized text to a file
    with open("transcribed_and_summarized_text.txt", "w", encoding="utf-8") as file:
        file.write("Transcribed Text:\n" + transcribed_text + "\n\n")
        file.write("Summarized Text:\n" + summarized_text+"\n\n")
        file.write("Bullet points:\n" + bullet_points)
    print("\nTranscribed and summarized text saved to 'transcribed_and_summarized_text.txt'")
else:
    print("Transcription error:", transcript.error)

