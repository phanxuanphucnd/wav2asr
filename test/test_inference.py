from arizona.asr.learner import Wav2AsrLearner

learner = Wav2AsrLearner(
    pretrain_model='path/to/pretrain.pt', 
    finetune_model='path/to/finetune.pt', 
    dictionary='path/to/dict.ltr.txt',
    lm_type='kenlm',
    lm_lexicon='path/to/lm/lexicon.txt', 
    lm_model='path/to/lm/lm.bin',
    lm_weight=1.5, 
    word_score=-1, 
    beam_size=50
)

hypos = learner.transcribe([
    './data/test_1.wav',
    './data/test_1.wav'
])

print("===")
print(hypos)