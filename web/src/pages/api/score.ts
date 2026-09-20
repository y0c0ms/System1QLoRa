import type { APIRoute } from 'astro';

export const prerender = false;

const SERVER = process.env.MODEL_SERVER ?? 'http://127.0.0.1:8900/score';

export const POST: APIRoute = async ({ request }) => {
  const body = await request.text();
  try {
    const upstream = await fetch(SERVER, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body,
    });
    return new Response(await upstream.text(), {
      status: upstream.status,
      headers: { 'Content-Type': 'application/json' },
    });
  } catch (err) {
    const message = err instanceof Error ? err.message : String(err);
    return new Response(JSON.stringify({ error: `model server unreachable: ${message}` }), {
      status: 502,
      headers: { 'Content-Type': 'application/json' },
    });
  }
};
