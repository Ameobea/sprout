import { browser } from '$app/environment';
import * as Sentry from '@sentry/browser';

const sentryEnabled = () => browser && !window.location.href.includes('localhost');

export const maybeInitSentry = () => {
  if (sentryEnabled()) {
    Sentry.init({
      dsn: 'https://0cbd7f7ac2ca48fe8fcc17cd8ba73c1d@sentry.ameo.design/12',
      integrations: [Sentry.browserTracingIntegration()],
      tracesSampleRate: 1.0,
    });
  }
};

export const getSentry = () => {
  if (!sentryEnabled()) {
    return null;
  }

  return Sentry;
};

export const captureMessage = (eventName: string, data?: any) =>
  getSentry()?.captureMessage(eventName, data ? { extra: data } : undefined);
