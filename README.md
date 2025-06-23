# MLX8-W3-VisionTransformer

├── README.md <-    
|   
├── /corpus    
│ 
├── ms marcos <- 
│ 
├── text8 <- 
│ 
├── checkpoints <- saved models
| 
├── requirements.txt <- The requirements file for reproducing the analysis environment, e.g. │ generated with pip freeze > requirements.txt 
│ 
└── 00_download_merge <- save ms marcos and text8 txt files 
| 
└── 01_tokenise_corpus.py <- combine into full corpus, preprocess, create indexes, lookups, inx_to_word and word_to_indx 
└── 02_doc_qry_data <- create two dicts: qrys {query_id: text, docs (id,id,id)} and docs {id:text}
└── 03_train_tt <- use dataloader, moniter via w&b, minimise triplet loss 
└── 04_server <- precompute doc embeddings using same tower, precache doc encodings, encode user query, use distance function to find and rank the top 5 docs 
| 
| 
└── dataset <- creates window and triplets functions 
└── eval_tt <- measure semantic similarity between a query and set of docs └── load_glove.py
└── models <- two symmetrical towers, one combined nn to combine outputs