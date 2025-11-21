import io
import os
import re
from collections import defaultdict
from typing import List, Tuple, Dict, Optional

import requests
import torch
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from PIL import Image
from transformers import CLIPProcessor, CLIPModel


# ------------------------------------------------------------------------------
# Model setup
# ------------------------------------------------------------------------------

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_NAME = "openai/clip-vit-base-patch32"  # 512-dim embeddings

clip_model = CLIPModel.from_pretrained(MODEL_NAME).to(DEVICE)
clip_processor = CLIPProcessor.from_pretrained(MODEL_NAME, use_fast=True)


# ------------------------------------------------------------------------------
# Attribute configuration (grouped by category)
# ------------------------------------------------------------------------------

ATTRIBUTE_CONFIG: Dict[str, List[Dict[str, str]]] = {
    "location": [
        {
            "id": "location.beachfront",
            "label": "beachfront hotel",
            "prompt": "a photo of a beachfront hotel",
        },
        {
            "id": "location.ocean_view",
            "label": "ocean view hotel",
            "prompt": "a photo of a hotel with ocean view",
        },
        {
            "id": "location.city_view",
            "label": "city view hotel",
            "prompt": "a photo of a hotel with city view",
        },
        {
            "id": "location.mountain_view",
            "label": "mountain view hotel",
            "prompt": "a photo of a hotel with mountain view",
        },
        {
            "id": "location.countryside",
            "label": "countryside hotel",
            "prompt": "a photo of a countryside hotel",
        },
    ],
    "exterior": [
        {
            "id": "exterior.day",
            "label": "hotel exterior day",
            "prompt": "a photo of a hotel exterior at day",
        },
        {
            "id": "exterior.night",
            "label": "hotel exterior night",
            "prompt": "a photo of a hotel exterior at night",
        },
        {
            "id": "exterior.entrance",
            "label": "hotel entrance",
            "prompt": "a photo of a hotel entrance",
        },
        {
            "id": "exterior.parking",
            "label": "hotel parking",
            "prompt": "a photo of a hotel parking lot",
        },
    ],
    "room": [
        {
            "id": "room.standard",
            "label": "standard room",
            "prompt": "a photo of a standard hotel room",
        },
        {
            "id": "room.deluxe",
            "label": "deluxe room",
            "prompt": "a photo of a deluxe hotel room",
        },
        {
            "id": "room.suite",
            "label": "suite",
            "prompt": "a photo of a hotel suite",
        },
        {
            "id": "room.family",
            "label": "family room",
            "prompt": "a photo of a family room with multiple beds",
        },
        {
            "id": "room.bathroom",
            "label": "bathroom",
            "prompt": "a photo of a hotel bathroom",
        },
        {
            "id": "room.balcony",
            "label": "room with balcony",
            "prompt": "a photo of a hotel room with balcony",
        },
    ],
    "amenities": [
        {
            "id": "pool.rooftop",
            "label": "rooftop pool",
            "prompt": "a photo of a rooftop pool",
        },
        {
            "id": "pool.outdoor",
            "label": "outdoor pool",
            "prompt": "a photo of an outdoor pool",
        },
        {
            "id": "pool.indoor",
            "label": "indoor pool",
            "prompt": "a photo of an indoor pool",
        },
        {
            "id": "pool.indoor_slides",
            "label": "indoor pool with slides",
            "prompt": "a photo of an indoor pool with slides",
        },
        {
            "id": "food.restaurant",
            "label": "restaurant",
            "prompt": "a photo of a hotel restaurant",
        },
        {
            "id": "food.breakfast",
            "label": "breakfast buffet",
            "prompt": "a photo of a hotel breakfast buffet",
        },
        {
            "id": "food.bar",
            "label": "bar",
            "prompt": "a photo of a hotel bar",
        },
        {
            "id": "public.lobby",
            "label": "lobby",
            "prompt": "a photo of a hotel lobby",
        },
        {
            "id": "wellness.gym",
            "label": "gym",
            "prompt": "a photo of a hotel gym",
        },
        {
            "id": "wellness.spa",
            "label": "spa",
            "prompt": "a photo of a hotel spa",
        },
        {
            "id": "family.kids_area",
            "label": "kids area",
            "prompt": "a photo of a hotel kids play area",
        },
        {
            "id": "business.center",
            "label": "business center",
            "prompt": "a photo of a hotel business center",
        },
    ],
}


def _flatten_attribute_config():
    """
    Turn ATTRIBUTE_CONFIG into:
      - FLAT_PROMPTS: list[str]
      - FLAT_META: list[dict(category, id, label, idx)]
      - CATEGORY_TO_INDICES: category -> list[int]
    """
    flat_prompts: List[str] = []
    flat_meta: List[Dict[str, object]] = []
    category_to_indices: Dict[str, List[int]] = defaultdict(list)

    for category, items in ATTRIBUTE_CONFIG.items():
        for item in items:
            idx = len(flat_prompts)
            flat_prompts.append(item["prompt"])
            flat_meta.append(
                {
                    "category": category,
                    "id": item["id"],
                    "label": item["label"],
                    "idx": idx,
                }
            )
            category_to_indices[category].append(idx)

    return flat_prompts, flat_meta, category_to_indices


FLAT_PROMPTS, FLAT_META, CATEGORY_TO_INDICES = _flatten_attribute_config()


# ------------------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------------------

def slugify(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9]+", "-", text)
    text = re.sub(r"-+", "-", text)
    return text.strip("-")


def guess_extension_from_content_type(content_type: str) -> str:
    """
    Guess extension from HTTP Content-Type.
    """
    mapping = {
        "image/jpeg": ".jpg",
        "image/jpg": ".jpg",
        "image/png": ".png",
        "image/webp": ".webp",
        "image/bmp": ".bmp",
        "image/tiff": ".tiff",
    }
    return mapping.get(content_type.lower(), ".jpg")


def guess_extension_from_url(url: str) -> str:
    """
    Fallback: inspect URL path for an extension.
    """
    path = url.split("?")[0]  # strip query
    _, ext = os.path.splitext(path)
    if ext:
        return ext
    return ".jpg"


# ------------------------------------------------------------------------------
# CLIP-based logic (per-category softmax, global-logit selection + gating)
# ------------------------------------------------------------------------------

def classify_hotel_attributes_pil(
    image: Image.Image,
    topk_per_category: int = 3,
    score_threshold: float = 0.05,
):
    """
    Per-category scoring:
    - One big CLIP call using all prompts at once.
    - Per-category softmax over that category's logits for within-category ranking.

    Returns:
        category_results: {
            "location": [ {label, category, prob, logit}, ... ],
            ...
        }
        logits_list: global logits for all prompts (for cross-category selection)
    """
    inputs = clip_processor(
        text=FLAT_PROMPTS,
        images=image.convert("RGB"),
        return_tensors="pt",
        padding=True,
    ).to(DEVICE)

    with torch.no_grad():
        outputs = clip_model(**inputs)
        logits_per_image = outputs.logits_per_image  # [1, num_prompts]
        logits = logits_per_image[0]  # [num_prompts]

    logits_list = logits.tolist()
    category_results: Dict[str, List[Dict[str, object]]] = {}

    for category, indices in CATEGORY_TO_INDICES.items():
        cat_logits = logits[indices]
        cat_probs = torch.softmax(cat_logits, dim=-1).tolist()

        entries: List[Dict[str, object]] = []
        for local_idx, prob in enumerate(cat_probs):
            if prob < score_threshold:
                continue
            global_idx = indices[local_idx]
            meta = FLAT_META[global_idx]
            entries.append(
                {
                    "label": meta["label"],
                    "category": category,
                    "prob": prob,                     # per-category probability
                    "logit": logits_list[global_idx], # global similarity
                }
            )

        # Sort within category by per-category prob, keep top-k
        entries.sort(key=lambda x: x["prob"], reverse=True)
        if entries:
            category_results[category] = entries[:topk_per_category]

    return category_results, logits_list


def generate_seo_filename_for_image(
    image: Image.Image,
    original_ext: str,
    topk_per_category: int = 3,
    score_threshold: float = 0.05,
    max_words: int = 10,
    max_total_labels: int = 5,
    logit_margin: float = 4.0,
    logit_floor: float = -5.0,
) -> Tuple[str, List[str]]:
    """
    Use per-category classification to generate a SEO-friendly filename.

    - Per-category softmax for internal ranking.
    - Global logits to decide which labels actually make the cut.
    - Dominant-category gating:
        * If the best label is in 'amenities', we drop 'room' labels from
          the SEO filename to avoid 'gym-deluxe-room' type nonsense.
    """
    category_results, logits_list = classify_hotel_attributes_pil(
        image=image,
        topk_per_category=topk_per_category,
        score_threshold=score_threshold,
    )

    # Collect all candidate labels with category info + global logit
    candidates_with_cat: List[Tuple[str, float, str]] = []
    for cat, entries in category_results.items():
        for e in entries:
            candidates_with_cat.append((e["label"], e["logit"], cat))

    if candidates_with_cat:
        # Best label overall (by global logit)
        best_label, best_logit, best_cat = max(
            candidates_with_cat, key=lambda x: x[1]
        )

        # Decide which categories are allowed based on dominant category
        allowed_categories = set(ATTRIBUTE_CONFIG.keys())
        if best_cat == "amenities":
            # If it's clearly an amenity (gym/spa/pool/etc.), don't call it a room
            allowed_categories.discard("room")

        # Apply global margin/floor filter on allowed categories only
        global_max = best_logit
        filtered_triplets: List[Tuple[str, float, str]] = [
            (label, logit, cat)
            for (label, logit, cat) in candidates_with_cat
            if cat in allowed_categories
            and logit >= global_max - logit_margin
            and logit >= logit_floor
        ]

        if not filtered_triplets:
            # Fallback: just use the single best label (if allowed)
            if best_cat in allowed_categories:
                chosen_labels = [best_label]
            else:
                chosen_labels = ["hotel"]
        else:
            filtered_triplets.sort(key=lambda x: x[1], reverse=True)
            chosen_labels = [
                label for (label, _logit, _cat) in filtered_triplets[:max_total_labels]
            ]
    else:
        chosen_labels = ["hotel"]

    phrase = " ".join(chosen_labels)
    words = phrase.split()
    phrase = " ".join(words[:max_words])

    slug = slugify(phrase)

    if not original_ext:
        original_ext = ".jpg"
    ext = original_ext.lower()

    filename = f"{slug}{ext}"
    return filename, chosen_labels


def get_image_embedding_pil(image: Image.Image) -> List[float]:
    """
    Return a 512-dim normalized CLIP embedding for an in-memory image.
    """
    inputs = clip_processor(
        images=image.convert("RGB"),
        return_tensors="pt"
    ).to(DEVICE)

    with torch.no_grad():
        image_features = clip_model.get_image_features(**inputs)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

    return image_features.cpu().tolist()[0]


# ------------------------------------------------------------------------------
# FastAPI setup
# ------------------------------------------------------------------------------

app = FastAPI(title="Hotel Image SEO & Embeddings Demo (Dominant-Category Gated)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # relax/tighten as needed
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ------------------------------------------------------------------------------
# Pydantic models
# ------------------------------------------------------------------------------

class UrlBatchRequest(BaseModel):
    urls: List[str]
    topk: int = 3                 # topk per category
    score_threshold: float = 0.05
    max_words: int = 10
    postfix: Optional[str] = None
    logit_margin: float = 4.0
    logit_floor: float = -5.0
    max_total_labels: int = 5


class UrlBatchResultItem(BaseModel):
    url: str
    seo_filename: Optional[str] = None
    seo_filename_with_postfix: Optional[str] = None
    error: Optional[str] = None


class UrlBatchResponse(BaseModel):
    results: List[UrlBatchResultItem]


class FilenameResponse(BaseModel):
    seo_filename: str
    seo_filename_with_postfix: Optional[str] = None
    attributes: Optional[List[str]] = None


class FilenameAndEmbeddingResponse(BaseModel):
    seo_filename: str
    seo_filename_with_postfix: Optional[str] = None
    attributes: Optional[List[str]] = None
    embedding: List[float]


# ------------------------------------------------------------------------------
# Endpoint 1: send an image, get SEO filename
# ------------------------------------------------------------------------------

@app.post("/filename-from-image", response_model=FilenameResponse)
async def filename_from_image(
    file: UploadFile = File(...),
    topk: int = 3,
    score_threshold: float = 0.05,
    max_words: int = 10,
    postfix: Optional[str] = None,
    max_total_labels: int = 5,
    logit_margin: float = 4.0,
    logit_floor: float = -5.0,
):
    """
    Input:  single uploaded image (multipart/form-data)
    Output: SEO filename (+ optional postfix) and attributes
    """
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))

        # Determine extension: prefer uploaded filename, fallback to content-type
        _, ext = os.path.splitext(file.filename or "")
        if not ext and file.content_type:
            ext = guess_extension_from_content_type(file.content_type)

        seo_filename, labels = generate_seo_filename_for_image(
            image=image,
            original_ext=ext,
            topk_per_category=topk,
            score_threshold=score_threshold,
            max_words=max_words,
            max_total_labels=max_total_labels,
            logit_margin=logit_margin,
            logit_floor=logit_floor,
        )

        seo_filename_with_postfix = f"{seo_filename}{postfix}" if postfix else None

        return FilenameResponse(
            seo_filename=seo_filename,
            seo_filename_with_postfix=seo_filename_with_postfix,
            attributes=labels,
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to process image: {e}")


# ------------------------------------------------------------------------------
# Endpoint 2: send multiple URLs, get JSON mapping URL -> new filename
# ------------------------------------------------------------------------------

@app.post("/filenames-from-urls", response_model=UrlBatchResponse)
async def filenames_from_urls(payload: UrlBatchRequest):
    """
    Input:  JSON with list of image URLs
    Output: for each URL, SEO filename (+ optional postfix) or error
    """
    results: List[UrlBatchResultItem] = []

    for url in payload.urls:
        try:
            resp = requests.get(url, timeout=10)
            if resp.status_code != 200:
                results.append(
                    UrlBatchResultItem(
                        url=url,
                        error=f"HTTP {resp.status_code}",
                    )
                )
                continue

            content_type = resp.headers.get("Content-Type", "").lower()
            ext = guess_extension_from_content_type(content_type)
            if ext == ".jpg":
                ext = guess_extension_from_url(url) or ".jpg"

            image = Image.open(io.BytesIO(resp.content))

            seo_filename, _labels = generate_seo_filename_for_image(
                image=image,
                original_ext=ext,
                topk_per_category=payload.topk,
                score_threshold=payload.score_threshold,
                max_words=payload.max_words,
                max_total_labels=payload.max_total_labels,
                logit_margin=payload.logit_margin,
                logit_floor=payload.logit_floor,
            )

            seo_filename_with_postfix = (
                f"{seo_filename}{payload.postfix}" if payload.postfix else None
            )

            results.append(
                UrlBatchResultItem(
                    url=url,
                    seo_filename=seo_filename,
                    seo_filename_with_postfix=seo_filename_with_postfix,
                )
            )
        except Exception as e:
            results.append(
                UrlBatchResultItem(
                    url=url,
                    error=f"Failed to process: {e}",
                )
            )

    return UrlBatchResponse(results=results)


# ------------------------------------------------------------------------------
# Endpoint 3: send an image, get SEO filename + embedding
# ------------------------------------------------------------------------------

@app.post("/filename-and-embedding", response_model=FilenameAndEmbeddingResponse)
async def filename_and_embedding(
    file: UploadFile = File(...),
    topk: int = 3,
    score_threshold: float = 0.05,
    max_words: int = 10,
    postfix: Optional[str] = None,
    max_total_labels: int = 5,
    logit_margin: float = 4.0,
    logit_floor: float = -5.0,
):
    """
    Input:  single uploaded image (multipart/form-data)
    Output: SEO filename (+ optional postfix), attributes, and CLIP embedding
    """
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))

        _, ext = os.path.splitext(file.filename or "")
        if not ext and file.content_type:
            ext = guess_extension_from_content_type(file.content_type)

        seo_filename, labels = generate_seo_filename_for_image(
            image=image,
            original_ext=ext,
            topk_per_category=topk,
            score_threshold=score_threshold,
            max_words=max_words,
            max_total_labels=max_total_labels,
            logit_margin=logit_margin,
            logit_floor=logit_floor,
        )

        embedding = get_image_embedding_pil(image)

        seo_filename_with_postfix = f"{seo_filename}{postfix}" if postfix else None

        return FilenameAndEmbeddingResponse(
            seo_filename=seo_filename,
            seo_filename_with_postfix=seo_filename_with_postfix,
            attributes=labels,
            embedding=embedding,
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to process image: {e}")


# ------------------------------------------------------------------------------
# Run with: uvicorn app:app --reload
# ------------------------------------------------------------------------------

