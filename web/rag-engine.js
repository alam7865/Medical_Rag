// In-browser retrieval engine for the static (Netlify) build.
//
// Mirrors src/rag/retriever.py: score every chunk by cosine similarity (embeddings are
// L2-normalized, so cosine == dot product), take the top `topK * multiplier` chunks, and
// collapse them to unique records keeping each record's best chunk.
// Pure functions only - no DOM, no network - so the same file runs in Node for tests.

export function decodeBase64Float32(b64) {
  let bytes;
  if (typeof atob === "function") {
    const bin = atob(b64);
    bytes = new Uint8Array(bin.length);
    for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
  } else {
    bytes = new Uint8Array(Buffer.from(b64, "base64"));
  }
  return new Float32Array(bytes.buffer, bytes.byteOffset, bytes.byteLength / 4);
}

/** Turn an exported index JSON (scripts/export_static_site.py) into a searchable index. */
export function loadIndex(payload) {
  const emb = decodeBase64Float32(payload.embeddings_b64);
  if (emb.length !== payload.n_chunks * payload.dim) {
    throw new Error(`Corrupt index ${payload.variant}: expected ${payload.n_chunks * payload.dim} floats, got ${emb.length}`);
  }
  return { variant: payload.variant, dim: payload.dim, n: payload.n_chunks, docs: payload.docs, chunkDoc: payload.chunk_doc, emb };
}

/**
 * @param index      result of loadIndex
 * @param query      Float32Array, L2-normalized query embedding
 * @param topK       number of unique records to return
 * @param multiplier chunks fetched = topK * multiplier (same as CHUNK_FETCH_MULTIPLIER)
 */
export function search(index, query, topK, multiplier = 3) {
  const { n, dim, emb, docs, chunkDoc } = index;
  if (query.length !== dim) throw new Error(`Query has ${query.length} dims, index has ${dim}`);

  const scores = new Float32Array(n);
  for (let c = 0; c < n; c++) {
    let s = 0;
    const off = c * dim;
    for (let j = 0; j < dim; j++) s += emb[off + j] * query[j];
    scores[c] = s;
  }
  const nFetch = Math.min(topK * multiplier, n);
  const order = Array.from({ length: n }, (_, i) => i).sort((a, b) => scores[b] - scores[a] || a - b).slice(0, nFetch);

  const out = [];
  const seen = new Set();
  for (const c of order) {
    const d = docs[chunkDoc[c]];
    if (seen.has(d.question_id)) continue;
    seen.add(d.question_id);
    out.push({
      rank: out.length + 1,
      question_id: d.question_id,
      question: d.question,
      answer: d.answer,
      answer_key: d.answer_key ?? "",
      score: Math.round(scores[c] * 1e6) / 1e6,
      cleaning_status: d.cleaning_status ?? null,
    });
    if (out.length === topK) break;
  }
  return out;
}
