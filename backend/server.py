"""
FastAPI Server for DocAI Knowledge Assistant.
Provides endpoints for document upload, management, individual deletion,
and SSE streaming chat with Claude-authentic dynamic thinking traces.
"""

import sys
import os
import re
import json
import time
import asyncio
from pathlib import Path
from typing import List, Optional, Generator
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.pipeline import DocAIPipeline
from src.generation.prompts import RAGPromptManager
from src.evaluation.evaluator import RAGEvaluator

app = FastAPI(title="DocAI Backend", version="1.0.0")

# Enable CORS for Next.js development server
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global pipeline instance
pipeline = DocAIPipeline()


class ChatRequest(BaseModel):
    message: str
    model: Optional[str] = "gemini-1.5-flash"
    api_key: Optional[str] = None


class KeyConfigRequest(BaseModel):
    gemini_api_key: Optional[str] = None


@app.get("/")
def root():
    """Root endpoint providing service overview, active model, and interactive documentation links."""
    return {
        "service": "DocAI Knowledge Assistant Backend API",
        "status": "online",
        "active_model": pipeline.llm.model_name,
        "provider": pipeline.llm.provider,
        "indexed_documents": len(pipeline.indexed_files),
        "docs_url": "/docs",
        "health_url": "/api/health",
        "frontend_url": "http://localhost:3000",
    }


@app.get("/api/health")
def health_check():
    """Returns engine health, indexed documents, active model, and Gemini API key status."""
    return {
        "status": "online",
        "model": pipeline.llm.model_name,
        "provider": pipeline.llm.provider,
        "has_gemini_key": bool(pipeline.llm.api_key and pipeline.llm.api_key.strip()),
        "indexed_documents_count": len(pipeline.indexed_files),
        "indexed_files": list(pipeline.indexed_files.values()),
    }


@app.get("/api/models")
def list_models():
    """Returns supported Gemini models and currently active selection."""
    return {
        "active_model": pipeline.llm.model_name,
        "supported_models": [
            {"id": "gemini-1.5-flash", "name": "Gemini 1.5 Flash", "desc": "Recommended • 1,500 req/day quota"},
            {"id": "gemini-2.0-flash", "name": "Gemini 2.0 Flash", "desc": "Next-Gen Flash • 1,500 req/day quota"},
            {"id": "gemini-1.5-pro", "name": "Gemini 1.5 Pro", "desc": "Deep reasoning • 50 req/day quota"},
            {"id": "gemini-2.5-flash", "name": "Gemini 2.5 Flash", "desc": "Experimental • 20 req/day quota"},
        ]
    }


@app.post("/api/keys")
def set_keys_endpoint(req: KeyConfigRequest):
    """Dynamically configures the Gemini API key on the active engine pipeline."""
    pipeline.llm.set_api_key(api_key=req.gemini_api_key)
    return {
        "status": "success",
        "has_gemini_key": bool(pipeline.llm.api_key),
        "provider": pipeline.llm.provider,
        "model": pipeline.llm.model_name,
    }


@app.get("/api/documents")
def list_documents():
    """Lists all indexed documents and their chunk statistics."""
    return {
        "count": len(pipeline.indexed_files),
        "documents": list(pipeline.indexed_files.values())
    }


@app.delete("/api/documents")
def clear_documents():
    """Clears all indexed documents from vector store and BM25 index."""
    pipeline.clear_all()
    return {"status": "cleared", "message": "All documents and memory cleared successfully"}


@app.delete("/api/documents/{filename}")
def delete_single_document(filename: str):
    """Deletes a specific document from vector store, BM25 index, and UI registry."""
    success = pipeline.delete_document(filename)
    if success:
        return {
            "status": "deleted",
            "filename": filename,
            "remaining_count": len(pipeline.indexed_files)
        }

    raise HTTPException(status_code=404, detail="Document not found")


@app.post("/api/upload")
async def upload_documents(files: List[UploadFile] = File(...)):
    """Uploads and indexes one or more documents (PDF, TXT, MD)."""
    uploaded_info = []

    for file in files:
        try:
            content = await file.read()
            import io
            file_stream = io.BytesIO(content)
            result = pipeline.ingest_file(file_stream, filename=file.filename)
            uploaded_info.append({
                "filename": file.filename,
                "status": "success",
                "total_chunks": result["metadata"]["total_chunks"],
                "total_pages": result["metadata"]["total_pages"],
                "size_bytes": len(content),
            })
        except Exception as e:
            uploaded_info.append({
                "filename": file.filename,
                "status": "error",
                "error": str(e)
            })

    return {"uploaded": uploaded_info, "total_indexed": len(pipeline.indexed_files)}


@app.post("/api/chat")
async def chat_endpoint(req: ChatRequest):
    """
    Streams SSE events for Claude-style dynamic thinking, real-time reasoning,
    answer tokens, and verified citations.
    """
    user_query = req.message.strip()
    if not user_query:
        raise HTTPException(status_code=400, detail="Message cannot be empty")

    # Update API key / model if provided
    if req.api_key is not None:
        pipeline.llm.set_api_key(api_key=req.api_key)

    if req.model and req.model.strip():
        pipeline.llm.set_model(req.model.strip())

    async def sse_event_stream():
        start_time = time.time()

        # Step 1: Thinking Start
        yield f"event: thinking_start\ndata: {json.dumps({'time': start_time})}\n\n"
        await asyncio.sleep(0.01)

        # Step 2: Contextualize Query & Retrieve
        recent_history = pipeline.memory.get_recent_history()
        search_query = pipeline.llm.contextualize_query(recent_history, user_query)

        t_ret_start = time.time()
        retrieved_chunks = pipeline.retriever.retrieve(search_query)
        retrieval_duration_ms = (time.time() - t_ret_start) * 1000

        # Step 3: Format Prompt
        prompt = RAGPromptManager.build_rag_prompt(user_query, retrieved_chunks)

        # Helper to smoothly pace answer tokens chunk-by-chunk for a natural typewriter animation
        async def stream_answer_clumps(text_chunk: str):
            if not text_chunk:
                return
            words = re.findall(r'\S+\s*', text_chunk)
            if not words:
                yield f"event: token\ndata: {json.dumps({'text': text_chunk})}\n\n"
                await asyncio.sleep(0.015)
                return
            i = 0
            while i < len(words):
                clump = "".join(words[i:i+2])
                i += 2
                yield f"event: token\ndata: {json.dumps({'text': clump})}\n\n"
                await asyncio.sleep(0.02)

        # Stream generation with robust state machine and buffering
        in_thought = False
        saw_thought_open = False
        saw_thought_close = False
        raw_stream_buffer = ""
        accumulated_answer = ""
        thought_buffer = ""
        t_gen_start = time.time()

        doc_names = ", ".join({c.metadata.get("source_file", "document") for c in retrieved_chunks})
        top_snippet = retrieved_chunks[0].text[:140].replace("\n", " ") if retrieved_chunks else ""

        for raw_chunk in pipeline.llm.stream_generate(
            prompt,
            user_query=user_query,
            retrieved_chunks=retrieved_chunks
        ):
            raw_stream_buffer += raw_chunk

            # If we haven't determined whether <thought> is present yet
            if not saw_thought_open:
                if "<thought>" in raw_stream_buffer:
                    saw_thought_open = True
                    in_thought = True
                    _, after_open = raw_stream_buffer.split("<thought>", 1)
                    raw_stream_buffer = after_open
                elif len(raw_stream_buffer) > 25 and not raw_stream_buffer.strip().startswith("<"):
                    # Model didn't use <thought> tags, synthesize dynamic query-specific contemplation first
                    saw_thought_open = True
                    saw_thought_close = True
                    dynamic_thought = (
                        f"Analyzing query: \"{user_query}\".\n\n"
                        f"Scanning knowledge base: Identified {len(retrieved_chunks)} relevant passage(s) across {doc_names or 'uploaded files'}.\n"
                        f"Analyzing key excerpt: \"{top_snippet}...\".\n\n"
                        f"Structuring verified response strictly grounded on document evidence."
                    )
                    for word in dynamic_thought.split(" "):
                        yield f"event: thinking_token\ndata: {json.dumps({'token': word + ' '})}\n\n"
                        await asyncio.sleep(0.015)
                    thinking_dur = round(time.time() - start_time, 2)
                    yield f"event: thinking_end\ndata: {json.dumps({'duration': thinking_dur})}\n\n"

                    ans_part = raw_stream_buffer
                    raw_stream_buffer = ""
                    accumulated_answer += ans_part
                    async for ev in stream_answer_clumps(ans_part):
                        yield ev
                    continue

            # If inside the thinking block: stream thinking tokens
            if in_thought:
                if "</thought>" in raw_stream_buffer:
                    thought_part, answer_part = raw_stream_buffer.split("</thought>", 1)
                    thought_buffer += thought_part
                    if thought_part:
                        yield f"event: thinking_token\ndata: {json.dumps({'token': thought_part})}\n\n"
                    in_thought = False
                    saw_thought_close = True
                    thinking_dur = round(time.time() - start_time, 2)
                    yield f"event: thinking_end\ndata: {json.dumps({'duration': thinking_dur})}\n\n"
                    raw_stream_buffer = ""
                    if answer_part.strip():
                        clean_ans = answer_part.lstrip()
                        accumulated_answer += clean_ans
                        async for ev in stream_answer_clumps(clean_ans):
                            yield ev
                else:
                    # Safely emit thought tokens while holding back enough characters to prevent splitting </thought>
                    if len(raw_stream_buffer) > 12:
                        to_yield = raw_stream_buffer[:-12]
                        raw_stream_buffer = raw_stream_buffer[-12:]
                        thought_buffer += to_yield
                        yield f"event: thinking_token\ndata: {json.dumps({'token': to_yield})}\n\n"
                        await asyncio.sleep(0.01)
            elif saw_thought_close:
                # In answer generation mode: yield smoothly paced tokens
                to_yield = raw_stream_buffer
                raw_stream_buffer = ""
                accumulated_answer += to_yield
                async for ev in stream_answer_clumps(to_yield):
                    yield ev

        # After LLM stream ends: flush remaining buffer
        if in_thought and not saw_thought_close:
            if "</thought>" in raw_stream_buffer:
                thought_part, answer_part = raw_stream_buffer.split("</thought>", 1)
                yield f"event: thinking_token\ndata: {json.dumps({'token': thought_part})}\n\n"
                clean_ans = answer_part.lstrip()
                accumulated_answer += clean_ans
            else:
                yield f"event: thinking_token\ndata: {json.dumps({'token': raw_stream_buffer})}\n\n"
            thinking_dur = round(time.time() - start_time, 2)
            yield f"event: thinking_end\ndata: {json.dumps({'duration': thinking_dur})}\n\n"
        elif saw_thought_close and raw_stream_buffer:
            accumulated_answer += raw_stream_buffer
            async for ev in stream_answer_clumps(raw_stream_buffer):
                yield ev
        elif not saw_thought_open and raw_stream_buffer:
            accumulated_answer += raw_stream_buffer
            async for ev in stream_answer_clumps(raw_stream_buffer):
                yield ev

        generation_duration_ms = (time.time() - t_gen_start) * 1000

        # Step 4: Evaluation Heuristics
        evaluation = RAGEvaluator.evaluate(
            query=user_query,
            answer=accumulated_answer,
            retrieved_chunks=retrieved_chunks,
            retrieval_ms=retrieval_duration_ms,
            generation_ms=generation_duration_ms,
        )

        citations_data = [
            {
                "source": c.metadata.get("source_file", "Unknown"),
                "page": c.metadata.get("page_number", 1),
                "chunk_id": c.chunk_id,
                "score": round(c.rrf_score, 4),
                "text_snippet": c.text[:220] + "..." if len(c.text) > 220 else c.text,
                "retrieval_method": c.retrieval_source,
            }
            for c in retrieved_chunks
        ]

        # Update Memory
        pipeline.memory.add_user_message(user_query)
        pipeline.memory.add_assistant_message(
            content=accumulated_answer,
            citations=citations_data,
            metrics={
                "retrieval_ms": round(retrieval_duration_ms, 1),
                "generation_ms": round(generation_duration_ms, 1),
                "relevance_score": evaluation.retrieval_relevance_score,
                "groundedness_score": evaluation.groundedness_score,
            }
        )

        # Step 5: Yield Verified Sources & Final Telemetry
        yield f"event: sources\ndata: {json.dumps({'citations': citations_data, 'evaluation': {'relevance': evaluation.retrieval_relevance_score, 'groundedness': evaluation.groundedness_score, 'citation_count': evaluation.citation_count, 'retrieval_ms': round(retrieval_duration_ms, 1), 'generation_ms': round(generation_duration_ms, 1), 'is_grounded': evaluation.is_grounded, 'summary': evaluation.diagnostic_summary}})}\n\n"
        yield f"event: done\ndata: {json.dumps({'status': 'complete'})}\n\n"

    return StreamingResponse(
        sse_event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.server:app", host="127.0.0.1", port=8000, reload=True)
