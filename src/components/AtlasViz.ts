import * as d3 from 'd3';
import type * as PIXI from 'pixi.js';
import type { Viewport } from 'pixi-viewport';

import type { MALUserAnimeListItem } from '../malAPI';
import type { EmbeddedPoint, Embedding } from '../routes/index';
import ColorLegend from './ColorLegend';

const WORLD_SIZE = 1;
const BASE_LABEL_FONT_SIZE = 42;
const BASE_RADIUS = 50;

enum ColorBy {
  AiredFromYear = 'aired_from_year',
  AverageRating = 'average_rating',
}

interface EmbeddedPointWithIndex extends EmbeddedPoint {
  index: number;
}

type EmbeddingWithIndices = EmbeddedPointWithIndex[];

export class AtlasViz {
  private embedding: EmbeddingWithIndices;
  private embeddedPointByID: Map<number, EmbeddedPoint>;
  private cachedNodeRadii: Float32Array;
  private colorBy = ColorBy.AiredFromYear;
  private colorScaler: d3.ScaleSequential<string, never>;
  private renderedHoverLabel: PIXI.Graphics | null = null;
  private setSelectedAnimeID: (id: number | null) => void;

  private PIXI: typeof import('pixi.js');
  private app: PIXI.Application;
  private container: Viewport;
  private pointsContainer: PIXI.Container;
  private labelsContainer: PIXI.Container;
  private textMeasurerCtx = (() => {
    const ctx = document.createElement('canvas').getContext('2d');
    ctx.font = '42px PT Sans';
    return ctx;
  })();

  public displayMALUser(allMALData: MALUserAnimeListItem[]) {
    const malData = allMALData.filter((d) => d.list_status.status !== 'plan_to_watch');

    // TODO: Clear styles on previously marked points

    // TODO: Update styles on user-owned points

    // TODO: Render backgrounds

    // TODO: Compute connections
  }

  private getColorByTitle() {
    switch (this.colorBy) {
      case ColorBy.AiredFromYear:
        return 'Aired from year';
      case ColorBy.AverageRating:
        return 'Average rating';
    }
  }

  private static createColorScaler = (scaleBy: ColorBy) => {
    switch (scaleBy) {
      case ColorBy.AiredFromYear:
        return d3.scaleSequential(d3.interpolatePlasma).domain([1990, 2022]);
      case ColorBy.AverageRating: {
        return d3.scaleSequential(d3.interpolateViridis).domain([3, 7.5]);
      }
    }
  };

  private static getNodeRadius = (ratingCount: number) => Math.pow(ratingCount, 0.8) / 8000 + 0.1825;

  /**
   * This is an override of the `uploadVertices` function for the original PIXI.js implementation.  It is optimized to
   * do less work given that
   */
  private customUploadVertices = (
    children: PIXI.Sprite[],
    _startIndex: number,
    amount: number,
    array: Float32Array | number[],
    stride: number,
    offset: number
  ) => {
    const adjustment = this.getNodeRadiusAdjustment(this.container.scale.x);

    const texture = children[0].texture;
    const orig = texture.orig;

    const w0 = orig.width; // * (1 - sprite.anchor.x);
    // const w1 = (orig.width); // * -sprite.anchor.x;
    const h0 = orig.height; // * (1 - sprite.anchor.y);
    // const h1 = orig.height; // * -sprite.anchor.y;

    for (let i = 0; i < amount; ++i) {
      const radius = this.cachedNodeRadii[i];
      const scale = (radius / BASE_RADIUS) * adjustment;

      // array[offset] = w1 * sx;
      // array[offset + 1] = h1 * sy;
      array[offset + stride] = w0 * scale;
      // array[offset + stride + 1] = h1 * sy;
      array[offset + stride * 2] = w0 * scale;
      array[offset + stride * 2 + 1] = h0 * scale;
      // array[offset + (stride * 3)] = w1 * scale;
      array[offset + stride * 3 + 1] = h0 * scale;
      offset += stride * 4;
    }
  };

  constructor(
    pixi: { PIXI: typeof import('pixi.js'); Viewport: typeof import('pixi-viewport').Viewport },
    containerID: string,
    embedding: Embedding,
    setSelectedAnimeID: (id: number | null) => void
  ) {
    this.embedding = embedding.map((datum, i) => ({ ...datum, index: i }));
    this.embeddedPointByID = new Map(embedding.map((p) => [+p.metadata.id, p]));
    this.setSelectedAnimeID = setSelectedAnimeID;
    this.PIXI = pixi.PIXI;

    // Performance optimization to avoid having to set transforms on every point
    this.cachedNodeRadii = new Float32Array(embedding.length);
    for (let i = 0; i < embedding.length; i++) {
      const p = embedding[i];
      const radius = AtlasViz.getNodeRadius(p.metadata.rating_count);
      this.cachedNodeRadii[i] = radius;
    }
    this.PIXI.ParticleRenderer.prototype.uploadVertices = this.customUploadVertices;

    const canvas = document.getElementById(containerID)! as HTMLCanvasElement;
    this.app = new this.PIXI.Application({
      antialias: true,
      resolution: window.devicePixelRatio,
      autoDensity: true,
      view: canvas,
      height: window.innerHeight,
      width: window.innerWidth,
      backgroundColor: 0,
    });

    this.container = new pixi.Viewport({
      screenWidth: window.innerWidth,
      screenHeight: window.innerHeight,
      worldHeight: WORLD_SIZE,
      worldWidth: WORLD_SIZE,
      interaction: this.app.renderer.plugins.interaction,
    });
    this.container.drag().pinch().wheel();
    // TODO: The initial transform probably needs to be relative to screen size
    this.container.setTransform(1000.5, 400.5, 16, 16);

    window.addEventListener('resize', () => {
      this.app.renderer.resize(window.innerWidth, window.innerHeight);
    });
    this.container.resize(window.innerWidth, window.innerHeight);

    this.pointsContainer = new this.PIXI.ParticleContainer(embedding.length, {
      vertices: true,
      position: false,
      rotation: false,
      uvs: false,
      tint: false,
    });

    this.pointsContainer.interactive = false;
    this.pointsContainer.interactiveChildren = false;
    this.container.addChild(this.pointsContainer);
    this.labelsContainer = new this.PIXI.Container();
    this.labelsContainer.interactive = false;
    this.labelsContainer.interactiveChildren = false;
    this.container.addChild(this.labelsContainer);

    // When zooming in and out, scale the circles in the opposite direction a bit to open up some space when
    // zooming in to dense areas and keeping structure when zooming out far.
    let lastScale = this.container.scale.x;

    this.app.ticker.add(() => {
      const newScale = this.container.scale.x;
      if (newScale === lastScale) {
        return;
      }
      lastScale = newScale;

      // || This is now computed in our overridden `uploadVertices` function to avoid the overhead
      // \/ of setting it on each node directly like this
      //
      // const adjustment = this.getNodeRadiusAdjustment(newScale);
      // this.pointsContainer.children.forEach((c, i) => {
      //   const radius = cachedNodeRadii[i];
      //   c.transform.scale.set((radius / BASE_RADIUS) * adjustment);
      // });

      this.labelsContainer.children.forEach((g) => {
        const d = (g as any).datum;
        this.setLabelScale(g as PIXI.Graphics, d, newScale);
      });
    });

    this.app.stage.addChild(this.container);

    // Somewhat annoyingly, we have to do manual hit testing in order to get decent rendering performance
    // for the circles.
    let hoveredDatum: EmbeddedPointWithIndex | null = null;

    canvas.addEventListener('pointermove', (evt) => {
      if (this.container.zooming) {
        return;
      }

      const worldPoint = this.container.toWorld(evt.offsetX, evt.offsetY);

      const newScale = this.container.scale.x;
      const adjustment = this.getNodeRadiusAdjustment(newScale);
      let datum: EmbeddedPointWithIndex | undefined;
      for (let i = this.embedding.length - 1; i >= 0; i--) {
        const p = this.embedding[i];
        const radius = this.cachedNodeRadii[i] * adjustment;
        const centerX = p.vector[0] + radius;
        const centerY = p.vector[1] + radius;
        const hitTest = Math.abs(worldPoint.x - centerX) < radius && Math.abs(worldPoint.y - centerY) < radius;

        if (hitTest) {
          datum = p;
          break;
        }
      }

      if (datum && hoveredDatum !== datum) {
        this.handlePointerOut();
        hoveredDatum = datum;
        this.handlePointerOver(hoveredDatum);
      } else if (!datum && hoveredDatum) {
        this.handlePointerOut();
        hoveredDatum = null;
      } else {
        hoveredDatum = datum;
      }

      this.container.cursor = hoveredDatum ? 'pointer' : 'grab';
    });

    let containerPointerDownPos: PIXI.Point | null = null;
    this.container
      .on('pointerdown', (evt: PIXI.InteractionEvent) => {
        this.container.cursor = 'grabbing';
        containerPointerDownPos = evt.data.getLocalPosition(this.app.stage);

        if (hoveredDatum) {
          this.handlePointerDown(hoveredDatum);
        }
      })
      .on('pointerup', (evt: PIXI.InteractionEvent) => {
        this.container.cursor = 'grab';
        const newPos = evt.data.getLocalPosition(this.app.stage);
        if (hoveredDatum || newPos.x !== containerPointerDownPos?.x || newPos.y !== containerPointerDownPos?.y) {
          containerPointerDownPos = null;
          return;
        }
        containerPointerDownPos = null;

        setSelectedAnimeID(null);
      });

    this.setColorBy(this.colorBy);

    this.renderPoints();

    this.renderLegend();
  }

  private getNodeRadiusAdjustment = (newZoomScale: number) => {
    let adjustment = (1 / (newZoomScale / 16)) * 1;
    adjustment = (adjustment + adjustment + adjustment + 1 + 1) / 5;
    adjustment = Math.min(adjustment, 1.5);
    return adjustment;
  };

  private handlePointerOver = (datum: EmbeddedPointWithIndex) => this.renderHoverLabel(datum);
  private handlePointerOut = () => this.maybeRemoveLabel();
  private handlePointerDown = (datum: EmbeddedPoint) => this.setSelectedAnimeID(datum.metadata.id);

  private renderPoints() {
    const { PIXI } = this;

    const nodeGraphics = new PIXI.Graphics();
    nodeGraphics.lineStyle(10, 0xffffff, 1);
    nodeGraphics.beginFill(0xffffff);
    nodeGraphics.drawCircle(0, 0, BASE_RADIUS);
    nodeGraphics.endFill();
    const texture = this.app.renderer.generateTexture(nodeGraphics, {
      resolution: 5,
      scaleMode: PIXI.SCALE_MODES.LINEAR,
      multisample: PIXI.MSAA_QUALITY.MEDIUM,
    });

    this.embedding.forEach((p) => {
      const { metadata, vector } = p;
      const [x, y] = vector;
      const radius = AtlasViz.getNodeRadius(metadata.rating_count);
      const nodeSprite = new PIXI.Sprite(texture);
      nodeSprite.interactive = false;
      nodeSprite.pivot.set(BASE_RADIUS, BASE_RADIUS);
      const colorString = this.colorScaler(metadata[this.colorBy]);
      const color = parseInt(colorString.slice(1), 16);
      nodeSprite.tint = color;
      nodeSprite.position.set(x, y);
      nodeSprite.scale.set(radius / BASE_RADIUS);
      (nodeSprite as any).radius = radius;
      (nodeSprite as any).datum = p;
      this.pointsContainer.addChild(nodeSprite);
    });
  }

  public setColorBy(colorBy: ColorBy) {
    this.colorBy = colorBy;
    this.colorScaler = AtlasViz.createColorScaler(colorBy);
    this.renderLegend();
  }

  public flyTo(id: number) {
    const [x, y] = this.embedding.find((p) => p.metadata.id === id)!.vector;
    this.container.animate({
      time: 500,
      position: { x, y },
      scale: 30,
      ease: (curTime: number, minVal: number, maxVal: number, maxTime: number): number => {
        // cubic ease in out
        const t = curTime / maxTime;
        const p = t * t * t;
        return minVal + (maxVal - minVal) * p;
      },
    });
  }

  private getTextScale = () => {
    const currentScale = this.container.scale.x;
    const textSize = (1 / currentScale) * 11 + 0.153;
    return textSize / BASE_LABEL_FONT_SIZE;
  };

  private setLabelScale = (g: PIXI.Graphics, datum: EmbeddedPointWithIndex, zoomScale: number) => {
    g.transform.scale.set(this.getTextScale());
    const radius = this.cachedNodeRadii[datum.index] * this.getNodeRadiusAdjustment(zoomScale);
    g.position.set(datum.vector[0] + radius, datum.vector[1]);
    const circleRadius = AtlasViz.getNodeRadius(datum.metadata.rating_count);
    g.position.y -= 12 * (1 / this.container.scale.x) + circleRadius * 0.0028 * this.container.scale.x;
  };

  private renderHoverLabel = (datum: EmbeddedPointWithIndex) => {
    this.maybeRemoveLabel();

    const text = datum.metadata.title;
    const textWidth = this.textMeasurerCtx.measureText(text).width;

    const g = new this.PIXI.Graphics();
    g.beginFill(0x222222, 0.5);
    g.drawRoundedRect(0, 0, textWidth + 10, 50, 5);
    g.endFill();
    g.interactive = false;
    g.interactiveChildren = false;

    (g as any).datum = datum;
    this.setLabelScale(g, datum, this.container.scale.x);
    // set origin to center of text
    g.pivot.set(textWidth / 2, 25);

    const textSprite = new this.PIXI.Text(text, {
      fontFamily: 'PT Sans',
      fontSize: BASE_LABEL_FONT_SIZE,
      fill: 0xcccccc,
      align: 'center',
    });
    textSprite.anchor.set(0.5, 0.5);
    textSprite.position.set(5 + textWidth / 2, 25);
    textSprite.interactive = false;

    g.addChild(textSprite);
    this.labelsContainer.addChild(g);
    this.renderedHoverLabel = g;
  };

  private maybeRemoveLabel = () => {
    const label = this.renderedHoverLabel;
    if (!label) {
      return;
    }

    label.parent?.removeChild(this.renderedHoverLabel);
    label.destroy({ children: true });
    this.renderedHoverLabel = null;
  };

  private renderLegend() {
    const legend = ColorLegend(this.colorScaler, {
      title: this.getColorByTitle(),
      tickFormat: (x) => x,
    });
    const legendContainer = document.getElementById('atlas-viz-legend')!;
    legendContainer.innerHTML = '';
    legendContainer.appendChild(legend);
  }

  public dispose() {
    this.app.ticker.stop();
  }
}
