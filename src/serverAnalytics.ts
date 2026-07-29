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

  void fetch(ANALYTICS_ENDPOINT, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      ...(meta.clientIP ? { 'X-Forwarded-For': meta.clientIP } : {}),
      ...(meta.userAgent ? { 'User-Agent': meta.userAgent } : {}),
    },
    body: JSON.stringify({ events: [event], verification, project: ANALYTICS_PROJECT }),
  }).catch(() => {
    // analytics must never break the app
  });
};
