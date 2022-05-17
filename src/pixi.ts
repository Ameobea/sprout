import { Viewport as Viewport_ } from 'pixi-viewport';
import * as gradients_ from '@pixi-essentials/gradients';

export * from '@pixi/constants';
export * from '@pixi/math';
export * from '@pixi/runner';
export * from '@pixi/settings';
export * from '@pixi/ticker';
export * from '@pixi/utils';
export * from '@pixi/display';
export * from '@pixi/core';
export * from '@pixi/events';
export * from '@pixi/particle-container';
export * from '@pixi/sprite';
export * from '@pixi/app';
export * from '@pixi/graphics';
export * from '@pixi/text';
export type { IHitArea } from '@pixi/interaction';
export * from '@pixi/interaction';

// Renderer plugins
import { Renderer } from '@pixi/core';
import { BatchRenderer } from '@pixi/core';
Renderer.registerPlugin('batch', BatchRenderer);
import { InteractionManager } from '@pixi/interaction';
Renderer.registerPlugin('interaction', InteractionManager);
import { ParticleRenderer } from '@pixi/particle-container';
Renderer.registerPlugin('particle', ParticleRenderer);

// Application plugins
import { Application } from '@pixi/app';
import { TickerPlugin } from '@pixi/ticker';
Application.registerPlugin(TickerPlugin);

export const Viewport = Viewport_;
export const gradients = gradients_;
