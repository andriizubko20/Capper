// Cloudflare Worker — проксі до api.sstats.net
// Деплой: dash.cloudflare.com → Workers & Pages → Create → Hello World → вставити цей код

const TARGET = 'https://api.sstats.net';

// Опційний shared secret — щоб random-сканери не били endpoint.
// Додай змінну PROXY_SECRET в Settings → Variables; клієнт має слати X-Proxy-Secret.
// Якщо PROXY_SECRET не задано — перевірка пропускається.

export default {
  async fetch(request, env) {
    if (env.PROXY_SECRET) {
      const provided = request.headers.get('x-proxy-secret') || '';
      if (provided !== env.PROXY_SECRET) {
        return new Response('forbidden', { status: 403 });
      }
    }

    const url = new URL(request.url);
    const target = `${TARGET}${url.pathname}${url.search}`;

    const headers = new Headers(request.headers);
    headers.delete('host');
    headers.delete('x-proxy-secret');

    const upstream = new Request(target, {
      method: request.method,
      headers,
      body: request.method === 'GET' || request.method === 'HEAD' ? undefined : request.body,
      redirect: 'follow',
    });

    try {
      const resp = await fetch(upstream);
      return new Response(resp.body, {
        status: resp.status,
        headers: resp.headers,
      });
    } catch (e) {
      return new Response(`proxy error: ${e.message}`, { status: 502 });
    }
  },
};
