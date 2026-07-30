import { createHash } from 'crypto';

import { dev } from '$app/environment';

import { ANALYTICS_ENDPOINT, ANALYTICS_PROJECT, ANALYTICS_SALT, type AnalyticsEvent } from 'src/analytics';

export const submitServerAnalyticsEvent = (
  event: AnalyticsEvent,
  meta: { clientIP?: string; userAgent?: string } = {}
): void => {
  if (dev) {
    console.debug('[server analytics]', event);
    return;
  }

  const verification = createHash('sha256')
    .update(event.category + event.subcategory + ANALYTICS_SALT)
    .digest('hex');

  // Daily-rotating synthetic session: groups a client's requests within a UTC day without
  // needing the client to carry any session state
  const sessionID = meta.clientIP
    ? createHash('sha256')
        .update(meta.clientIP + (meta.userAgent ?? '') + new Date().toISOString().slice(0, 10) + ANALYTICS_SALT)
        .digest('hex')
        .slice(0, 16)
    : undefined;

  void fetch(ANALYTICS_ENDPOINT, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      ...(meta.clientIP ? { 'X-Forwarded-For': meta.clientIP } : {}),
      ...(meta.userAgent ? { 'User-Agent': meta.userAgent } : {}),
    },
    body: JSON.stringify({
      events: [event],
      verification,
      project: ANALYTICS_PROJECT,
      ...(sessionID ? { session_id: sessionID } : {}),
    }),
  }).catch(() => {
    // analytics must never break the app
  });
};
