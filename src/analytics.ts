import { browser } from '$app/env';

export interface AnalyticsEvent {
  category: string;
  subcategory: string;
  payload?: unknown;
}

export const ANALYTICS_ENDPOINT = 'https://osu-api-bridge.ameo.dev/a/z';
export const ANALYTICS_SALT = '4rW9XKHcEKa6bolWry8k0LGW';
export const ANALYTICS_PROJECT = 'sprout-legacy';

const analyticsEnabled = () => browser && !window.location.href.includes('localhost');

const computeVerificationHash = async (events: AnalyticsEvent[]): Promise<string> => {
  const encoder = new TextEncoder();
  const hashBuffer = await crypto.subtle.digest(
    'SHA-256',
    encoder.encode(events.map((evt) => evt.category + evt.subcategory).join('') + ANALYTICS_SALT)
  );
  return Array.from(new Uint8Array(hashBuffer))
    .map((b) => b.toString(16).padStart(2, '0'))
    .join('');
};

let sessionID: string | null = null;
const getSessionID = (): string => {
  if (sessionID) {
    return sessionID;
  }
  const gen = () => {
    const bytes = crypto.getRandomValues(new Uint8Array(8));
    return Array.from(bytes)
      .map((b) => b.toString(16).padStart(2, '0'))
      .join('');
  };
  try {
    sessionID = sessionStorage.getItem('analyticsSessionID');
    if (!sessionID) {
      sessionID = gen();
      sessionStorage.setItem('analyticsSessionID', sessionID);
    }
  } catch (_err) {
    sessionID = gen();
  }
  return sessionID;
};

const queue: AnalyticsEvent[] = [];
let flushTimer: ReturnType<typeof setTimeout> | null = null;

const flush = async () => {
  const events = queue.splice(0);
  if (!events.length) {
    return;
  }
  try {
    const verification = await computeVerificationHash(events);
    await fetch(ANALYTICS_ENDPOINT, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        events,
        verification,
        project: ANALYTICS_PROJECT,
        session_id: getSessionID(),
      }),
      keepalive: true,
    });
  } catch (_err) {
    // analytics must never break the app
  }
};

if (browser) {
  window.addEventListener('pagehide', () => {
    if (queue.length) {
      void flush();
    }
  });
}

export const maybeInitAnalytics = () => {};

const SubmittedOnceEvents = new Set<string>();

export const submitAnalyticsEvent = (event: AnalyticsEvent, once = false): void => {
  if (!browser) {
    return;
  }
  if (!analyticsEnabled()) {
    console.debug('[analytics]', event);
    return;
  }

  if (once) {
    const eventID = `${event.category}::${event.subcategory}`;
    if (SubmittedOnceEvents.has(eventID)) {
      return;
    }
    SubmittedOnceEvents.add(eventID);
  }

  queue.push(event);
  // If the page is being hidden/unloaded, timers won't fire; flush immediately via keepalive fetch
  if (document.visibilityState === 'hidden') {
    void flush();
    return;
  }
  if (flushTimer === null) {
    flushTimer = setTimeout(() => {
      flushTimer = null;
      void flush();
    }, 800);
  }
};

export type AnalyticsSurface = 'user_profile' | 'interactive' | 'standalone';

export const getSurface = (): AnalyticsSurface => {
  if (!browser) {
    return 'standalone';
  }
  const p = window.location.pathname;
  if (p.startsWith('/user/')) {
    return 'user_profile';
  }
  if (p.startsWith('/interactive-recommender')) {
    return 'interactive';
  }
  return 'standalone';
};

// Distinguishes SSR pages reached via in-app navigation (a real user interaction) from direct
// URL hits/crawlers, which must not produce events.
let inAppNav = false;

export const markInAppNav = () => {
  inAppNav = true;
};

export const wasInAppNav = (): boolean => inAppNav;
