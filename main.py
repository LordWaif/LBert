import os
os.chdir(__file__.rsplit("/",1)[0])
from transformers import BertTokenizer
from LBert.classifier import CustomBertClassifier
import datasets
from datasets import load_dataset
from config import PRE_TRAINED_MODEL_NAME,EPOCHS,PREDICT_POOLER,BATCH_SIZE,MAX_LENGTH,OVERLAP,MAX_LENGTH_TOKENS,LR,LOSS,OPTIMIZER
from config import ACCUMULATIVE_STEPS
import torch
from train_eval_fn import train_epoch,eval_model
if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    from LBert.dataset import CustomDataset,CustomBatchSampler,custom_collate_fn
    from transformers import get_linear_schedule_with_warmup
    import json,numpy as np
    labels_json = json.load(open("labels.json"))
    ecthr_cases = load_dataset("ecthr_cases",data_dir="alleged-violation-prediction")
    train = ecthr_cases["train"] # type: ignore
    test = ecthr_cases["test"] # type: ignore
    validation = ecthr_cases["validation"] # type: ignore
    model = CustomBertClassifier(PRE_TRAINED_MODEL_NAME,PREDICT_POOLER,num_classes=len(labels_json))
    model = model.to(device)
    tokenizer = BertTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME)
    # REAL BATCH_SIZE SUPERIOR = batch_size*(max_length_tokens//max_length)

    def one_hot_encoding(labels, label_map):
        num_labels = len(label_map)
        one_hot = np.zeros(num_labels)
        for label in labels:
            label_index = label_map.get(label)
            if label_index is not None:
                one_hot[label_index] = 1
        return one_hot
    
    def createDataLoader(data:datasets.Dataset,max_length,overlap,max_length_tokens,batch_size,tokenizer,labels_json):
        labels = [one_hot_encoding(lb,labels_json) for lb in train['labels']] # type: ignore
        text = data['facts']
        dataset = CustomDataset(text,labels,tokenizer,max_length=max_length,overlap=overlap,max_length_tokens=max_length_tokens)
        dataloader = dataset.toDataLoader(
            collate_fn=custom_collate_fn,
            batch_sampler=CustomBatchSampler(dataset,batch_size=batch_size)
        )
        return dataloader
    
    train_dataloader = createDataLoader(train,MAX_LENGTH,OVERLAP,MAX_LENGTH_TOKENS,BATCH_SIZE,tokenizer,labels_json) # type: ignore
    test_dataloader = createDataLoader(test,MAX_LENGTH,OVERLAP,MAX_LENGTH_TOKENS,BATCH_SIZE,tokenizer,labels_json) # type: ignore
    validation_dataloader = createDataLoader(validation,MAX_LENGTH,OVERLAP,MAX_LENGTH_TOKENS,BATCH_SIZE,tokenizer,labels_json) # type: ignore

    optimizer = OPTIMIZER(model.parameters(),lr=LR,correct_bias=False)
    total_steps = len(train_dataloader) * EPOCHS

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=total_steps
    )

    loss_fn = LOSS().to(device)
    history = []
    best_acc = 0

    from LBert.dataset import EarlyStopping
    early_stopping = EarlyStopping(patience=5, verbose=True)

    from accelerate import Accelerator
    accelerator = Accelerator(gradient_accumulation_steps=ACCUMULATIVE_STEPS)

    model, optimizer, train_dataloader, scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, scheduler
    )
    for epoch in range(EPOCHS):
        print(f'Epoch {epoch+1}/{EPOCHS}')

        classification_report_train = train_epoch(
            model,
            train_dataloader,
            loss_fn,
            optimizer,
            device,
            scheduler,
            accelerator,
            labels_name = list(labels_json.keys()),
        )

        classification_report_eval = eval_model(
            model,
            validation_dataloader,
            loss_fn,
            device,
            early_stopping=early_stopping,
            labels_name = list(labels_json.keys())
        )
        history.append({
            'epoch': epoch+1,
            'train_cr': classification_report_train,
            'eval_cr': classification_report_eval,
            'eval_loss': classification_report_eval['loss']
        })

        print(f'Eval loss {classification_report_eval["loss"]}')

    classification_report_test = eval_model(
        model,
        test_dataloader,
        loss_fn,
        device,
        labels_name = list(labels_json.keys())
    )

    history = {'train':history,'test':classification_report_test}
    json.dump(history,open('history.json','w'),ensure_ascii=False,indent=4)


    


    # one_hot_encoding_labels = {l:_ for _,l in enumerate(sorted(set([_ for l in ecthr_cases["train"]["labels"] for _ in l]),key=lambda x: (0,x) if x.startswith("P") else (1,x)))} # type: ignore
    # import json
    # with open("labels.json","w") as f:
    #     json.dump(one_hot_encoding_labels,f,indent=4)
