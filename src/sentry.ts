import { browser } from '$app/environment';
import * as Sentry from '@sentry/browser';

const sentryEnabled = () => browser && !window.location.href.includes('localhost');

export const maybeInitSentry = () => {
  if (sentryEnabled()) {
    Sentry.init({ dsn: 'https://e0edc2ad30c53e2b98f380341d285871@sentry.ameo.design/12' });
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
