# imgclass
Image classification demo running on RTX4000 ada

- Creates a server running on port 8000 using uvicorn.
    - endpoint: POST: /filename-from-image
        parameter: file - attached image to classify
        parameter: topk - how many attributes to consider
        parameter: score_threshold - minimum probability cutoff
        parameter: max_words - max keywords in the filename

    - endpoint: POST: /filenames-from-urls
        body: JSON-formatted. include topk, score_threshold, and max_words as above (in body).
           detail to be added. 

    - endpoint: POST: /filename-and-embedding - returns proposed filename and vector embeddings
        parameters: as above.
