import * as d3 from 'd3';
import type * as PIXI from 'pixi.js';
import type { Viewport } from 'pixi-viewport';

import type { MALUserAnimeListItem } from '../malAPI';
import type { EmbeddedPoint, Embedding } from '../routes/index';
import ColorLegend from './ColorLegend';

const WORLD_SIZE = 1;
const BASE_LABEL_FONT_SIZE = 42;
const BASE_RADIUS = 50;
const MAL_NODE_COLOR = 0x2bcaff;
const SELECTED_NODE_COLOR = 0xdb18ce;
const NEIGHBOR_LINE_COLOR = 0xefefef;
const NEIGHBOR_LINE_OPACITY = 0.4;
/**
 * Time to force a label to stay rendered after first rendering it to avoid labels flickering in and out
 * rapidly due to the way the labeling algorithm works.
 */
const LABEL_PIN_TIME_MS = 750;
const LABEL_BG_COLOR = 0x080808;
const LABEL_BG_OPACITY = 0.82;
const LABEL_HEIGHT = 50;
const LABEL_SCALE_MULTIPLIER = 0.9;

enum ColorBy {
  AiredFromYear = 'aired_from_year',
  AverageRating = 'average_rating',
}

interface EmbeddedPointWithIndex extends EmbeddedPoint {
  index: number;
}

type EmbeddingWithIndices = EmbeddedPointWithIndex[];

const buildNodeLabelClass = (PixiModule: typeof PIXI) => {
  class NodeLabel extends PixiModule.Graphics {
    constructor(text: string, style: Partial<PIXI.ITextStyle>, textWidth: number) {
      super();
      this.beginFill(LABEL_BG_COLOR, LABEL_BG_OPACITY);
      const paddingHorizontal = 4;
      const paddingVertical = 2;
      this.drawRoundedRect(
        -textWidth / 2 - paddingHorizontal / 2,
        -LABEL_HEIGHT / 2 - paddingVertical / 2,
        textWidth + paddingHorizontal,
        LABEL_HEIGHT + paddingVertical,
        5
      );
      this.textNode = new PixiModule.Text(text, style);
      this.textNode.position.set(-textWidth / 2, -LABEL_HEIGHT / 2);
      this.addChild(this.textNode);
    }

    private textNode: PIXI.Text;
    public renderTimeMs: number;
    public nodeIx: number;
    public datum: EmbeddedPointWithIndex;
    public textWidth: number;
  }

  return NodeLabel;
};

type InstanceTypeOf<T> = T extends new (...args: any[]) => infer R ? R : never;

type NodeLabelClass = ReturnType<typeof buildNodeLabelClass>;
// Instance type
type NodeLabel = InstanceTypeOf<NodeLabelClass>;

export class AtlasViz {
  private embedding: EmbeddingWithIndices;
  private embeddedPointByID: Map<number, EmbeddedPointWithIndex>;
  /**
   * Node positions from embedding kept here for faster lookup
   */
  private embeddingPositions: Float32Array;
  private cachedNodeRadii: Float32Array;
  private colorBy = ColorBy.AiredFromYear;
  private colorScaler: d3.ScaleSequential<string, never>;
  private renderedHoverObjects: { label: PIXI.Graphics; neighborLines: PIXI.Graphics | null } | null = null;
  private neighbors: number[][] = null;
  private setSelectedAnimeID: (id: number | null) => void;

  private PIXI: typeof import('pixi.js');
  private gradients: typeof import('@pixi-essentials/gradients');
  private app: PIXI.Application;
  private container: Viewport;
  private pointsContainer: PIXI.Container;
  private selectedNodeContainer: PIXI.Container;
  private decorationsContainer: PIXI.Container;
  private labelsContainer: PIXI.Container;
  private hoverLabelsContainer: PIXI.Container;
  private textMeasurerCtx = (() => {
    const ctx = document.createElement('canvas').getContext('2d');
    ctx.font = '42px PT Sans';
    return ctx;
  })();
  private malPointBackgrounds: PIXI.ParticleContainer | null = null;
  private cachedMALBackgroundTexture: PIXI.Texture | null = null;
  private renderedMALNodeIDs: Set<number> = new Set();
  private selectedNode: {
    id: number;
    node: PIXI.Sprite;
    background: PIXI.Sprite;
    connections: PIXI.Graphics | null;
  } | null = null;
  private cachedNodeTexture: PIXI.Texture | null = null;
  private cachedLabels: Map<string, NodeLabel> = new Map();
  private visibleNodesScratch: Uint32Array;
  private visibleNodesQuadIndices: Uint32Array;
  private labelUnrenderTimes: Float64Array;
  // innerWidth * innerHeight
  private screenSpacePx: number;
  private NodeLabel: NodeLabelClass;

  private buildNeighborLines(datum: EmbeddedPointWithIndex): PIXI.Graphics | null {
    if (!this.neighbors) {
      return null;
    }

    const neighbors: number[] = this.neighbors[datum.index] ?? [];
    if (neighbors.length === 0) {
      console.warn(`No neighbors found for node index=${datum.index} id=${datum.metadata.id}`);
      return null;
    }

    const g = new this.PIXI.Graphics();
    g.lineStyle(2, NEIGHBOR_LINE_COLOR, NEIGHBOR_LINE_OPACITY, 0.5, true);

    neighbors.forEach((neighborID) => {
      const neighbor = this.embeddedPointByID.get(neighborID);
      if (!neighbor) {
        console.warn(`Could not find neighbor id=${neighborID} for node index=${datum.index}`);
        return;
      }
      g.moveTo(datum.vector.x, datum.vector.y);
      g.lineTo(neighbor.vector.x, neighbor.vector.y);
    });
    return g;
  }

  private getNodeBackgroundTexture = () => {
    if (this.cachedMALBackgroundTexture) {
      return this.cachedMALBackgroundTexture;
    }

    const gradientRenderTexture = this.gradients.GradientFactory.createRadialGradient(
      this.app.renderer as PIXI.Renderer,
      this.PIXI.RenderTexture.create({ width: BASE_RADIUS * 2, height: BASE_RADIUS * 2 }),
      {
        x0: BASE_RADIUS,
        y0: BASE_RADIUS,
        r0: 0,
        x1: BASE_RADIUS,
        y1: BASE_RADIUS,
        r1: BASE_RADIUS,
        colorStops: [
          { color: 0xffffff40, offset: 0 },
          { color: 0xffffff34, offset: 0.1 },
          { color: 0xffffff28, offset: 0.3 },
          { color: 0xffffff04, offset: 0.6 },
          { color: 0xffffff00, offset: 0.96 },
        ],
      }
    );

    this.cachedMALBackgroundTexture = gradientRenderTexture;
    return gradientRenderTexture;
  };

  private buildNodeBackgroundSprite = (
    texture: PIXI.Texture,
    datum: EmbeddedPointWithIndex,
    color: number
  ): PIXI.Sprite => {
    const nodeRadius = this.cachedNodeRadii[datum.index];
    const radius = 5.8 + 0.8 * nodeRadius;
    const sprite = new this.PIXI.Sprite(texture);
    sprite.blendMode = this.PIXI.BLEND_MODES.COLOR;
    sprite.interactive = false;
    sprite.tint = color;
    sprite.position.set(datum.vector.x, datum.vector.y);
    sprite.anchor.set(0.5, 0.5);
    sprite.scale.set(radius / BASE_RADIUS);
    return sprite;
  };

  public displayMALUser(allMALData: MALUserAnimeListItem[]) {
    const malData = allMALData.filter((d) => d.list_status.status !== 'plan_to_watch');

    this.renderedMALNodeIDs.clear();
    malData.forEach((d) => this.renderedMALNodeIDs.add(d.node.id));
    this.renderNodes();

    if (this.malPointBackgrounds) {
      this.decorationsContainer.removeChild(this.malPointBackgrounds);
      this.malPointBackgrounds.destroy({ children: true });
      this.malPointBackgrounds = null;
    }
    this.malPointBackgrounds = new this.PIXI.ParticleContainer(allMALData.length, {
      vertices: false,
      position: false,
      rotation: false,
      uvs: false,
      tint: false,
      alpha: false,
      scale: false,
    });

    const texture = this.getNodeBackgroundTexture();
    malData.forEach((item) => {
      // TODO: Dynamic color based on rating?
      const color = MAL_NODE_COLOR;
      const datum = this.embeddedPointByID.get(+item.node.id);
      if (!datum) {
        console.warn(`Could not find embedded point for MAL data point ${item.node.id} (${item.node.title})`);
        return;
      }

      const sprite = this.buildNodeBackgroundSprite(texture, datum, color);
      this.malPointBackgrounds.addChild(sprite);
    });
    this.decorationsContainer.addChild(this.malPointBackgrounds);

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

  private static getNodeRadius = (ratingCount: number) => Math.pow(ratingCount, 0.78) / 9000 + 0.14;

  private getNodeColor = (datum: EmbeddedPoint) => {
    const animeID = datum.metadata.id;
    if (this.renderedMALNodeIDs.has(animeID)) {
      return MAL_NODE_COLOR;
    }

    const colorString = this.colorScaler(datum.metadata[this.colorBy]);
    const color = parseInt(colorString.slice(1), 16);
    return color;
  };

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

    const w0 = orig.width * (1 - 0.5); // * (1 - sprite.anchor.x);
    const w1 = orig.width * -0.5;
    const h0 = orig.height * (1 - 0.5); // * (1 - sprite.anchor.y);
    const h1 = orig.height * -0.5;

    for (let i = 0; i < amount; ++i) {
      const radius = this.cachedNodeRadii[i];
      const scale = (radius / BASE_RADIUS) * adjustment;

      array[offset] = w1 * scale;
      array[offset + 1] = h1 * scale;
      array[offset + stride] = w0 * scale;
      array[offset + stride + 1] = h1 * scale;
      array[offset + stride * 2] = w0 * scale;
      array[offset + stride * 2 + 1] = h0 * scale;
      array[offset + stride * 3] = w1 * scale;
      array[offset + stride * 3 + 1] = h0 * scale;
      offset += stride * 4;
    }
  };

  constructor(
    pixi: {
      PIXI: typeof import('pixi.js');
      Viewport: typeof import('pixi-viewport').Viewport;
      gradients: typeof import('@pixi-essentials/gradients');
    },
    containerID: string,
    embedding: Embedding,
    setSelectedAnimeID: (id: number | null) => void
  ) {
    this.embedding = embedding.map((datum, i) => ({ ...datum, index: i }));
    this.embeddingPositions = new Float32Array(embedding.length * 2);
    for (let i = 0; i < embedding.length; i++) {
      this.embeddingPositions[i * 2] = embedding[i].vector.x;
      this.embeddingPositions[i * 2 + 1] = embedding[i].vector.y;
    }
    this.visibleNodesScratch = new Uint32Array(embedding.length);
    this.visibleNodesQuadIndices = new Uint32Array(embedding.length);
    this.labelUnrenderTimes = new Float64Array(embedding.length);
    this.embeddedPointByID = new Map(this.embedding.map((p) => [+p.metadata.id, p]));
    this.setSelectedAnimeID = (newSelectedAnimeID: number | null) => {
      setSelectedAnimeID(newSelectedAnimeID);

      if (this.selectedNode?.id !== newSelectedAnimeID) {
        const sprites = this.renderSelectedNodeObjects(newSelectedAnimeID);
        this.selectedNode = sprites ? { id: newSelectedAnimeID, ...sprites } : null;
      }
    };
    this.PIXI = pixi.PIXI;
    this.gradients = pixi.gradients;
    this.NodeLabel = buildNodeLabelClass(this.PIXI);

    // Performance optimization to avoid having to set transforms on every point
    this.cachedNodeRadii = new Float32Array(embedding.length);
    for (let i = 0; i < embedding.length; i++) {
      const p = embedding[i];
      const radius = AtlasViz.getNodeRadius(p.metadata.rating_count);
      this.cachedNodeRadii[i] = radius;
    }

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
    this.container.drag({ mouseButtons: 'middle-left' }).pinch().wheel();
    // TODO: The initial transform probably needs to be relative to screen size
    this.container.setTransform(1000.5, 400.5, 8, 8);

    this.screenSpacePx = window.innerWidth * window.innerHeight;
    window.addEventListener('resize', () => {
      this.app.renderer.resize(window.innerWidth, window.innerHeight);
      this.container.resize(window.innerWidth, window.innerHeight);
      this.screenSpacePx = window.innerWidth * window.innerHeight;
    });

    // Need to do some hacky subclassing to enable big performance improvement
    class NodesParticleRenderer extends this.PIXI.ParticleRenderer {}
    NodesParticleRenderer.prototype.uploadVertices = this.customUploadVertices;

    const nodesParticleRenderer = new NodesParticleRenderer(this.app.renderer as PIXI.Renderer);

    class NodesParticleContainer extends this.PIXI.ParticleContainer {
      public render(renderer: PIXI.Renderer): void {
        if (!this.visible || this.worldAlpha <= 0 || !this.children.length || !this.renderable) {
          return;
        }

        renderer.batch.setObjectRenderer(nodesParticleRenderer);
        nodesParticleRenderer.render(this);
      }
    }

    this.decorationsContainer = new this.PIXI.Container();
    this.decorationsContainer.interactive = false;
    this.decorationsContainer.interactiveChildren = false;
    this.container.addChild(this.decorationsContainer);

    this.pointsContainer = new NodesParticleContainer(embedding.length, {
      vertices: true,
      position: false,
      rotation: false,
      uvs: false,
      tint: false,
    });
    this.pointsContainer.interactive = false;
    this.pointsContainer.interactiveChildren = false;
    this.container.addChild(this.pointsContainer);

    this.selectedNodeContainer = new this.PIXI.Container();
    this.selectedNodeContainer.interactive = false;
    this.selectedNodeContainer.interactiveChildren = false;
    this.container.addChild(this.selectedNodeContainer);

    this.labelsContainer = new this.PIXI.Container();
    this.labelsContainer.interactive = false;
    this.labelsContainer.interactiveChildren = false;
    this.container.addChild(this.labelsContainer);

    this.hoverLabelsContainer = new this.PIXI.Container();
    this.hoverLabelsContainer.interactive = false;
    this.hoverLabelsContainer.interactiveChildren = false;
    this.container.addChild(this.hoverLabelsContainer);

    // When zooming in and out, scale the circles in the opposite direction a bit to open up some space when
    // zooming in to dense areas and keeping structure when zooming out far.
    let lastCenter = this.container.center;
    let lastScale = this.container.scale.x;

    this.app.ticker.add(() => {
      const newScale = this.container.scale.x;

      if (
        lastCenter.x === this.container.center.x &&
        lastCenter.y === this.container.center.y &&
        newScale === lastScale
      ) {
        return;
      }

      if (newScale !== lastScale) {
        const adjustment = this.getNodeRadiusAdjustment(newScale);
        this.labelsContainer.children.forEach((label: NodeLabel) =>
          this.setLabelScale(label, label.datum, adjustment, LABEL_SCALE_MULTIPLIER)
        );
      }

      // If we are only zooming, clearing pins helps looks better
      const clearPins = newScale !== lastScale;
      lastCenter = this.container.center;
      this.updateLabels(clearPins);

      if (newScale === lastScale) {
        return;
      }
      lastScale = newScale;

      const adjustment = this.getNodeRadiusAdjustment(newScale);

      // || This is now computed in our overridden `uploadVertices` function to avoid the overhead
      // \/ of setting it on each node directly like this
      //
      // this.pointsContainer.children.forEach((c, i) => {
      //   const radius = this.cachedNodeRadii[i];
      //   c.transform.scale.set((radius / BASE_RADIUS) * adjustment);
      // });

      if (this.selectedNode) {
        const index = this.embeddedPointByID.get(this.selectedNode.id)!.index;
        const radius = this.cachedNodeRadii[index];
        this.selectedNode.node.transform.scale.set((radius / BASE_RADIUS) * adjustment);
      }

      this.hoverLabelsContainer.children.forEach((g) => {
        const d = (g as any).datum;
        this.setLabelScale(g as PIXI.Graphics, d, adjustment);
      });
    });

    this.app.stage.addChild(this.container);

    // Somewhat annoyingly, we have to do manual hit testing in order to get decent rendering performance
    // for the circles.
    let hoveredDatum: EmbeddedPointWithIndex | null = null;
    let containerPointerDownPos: PIXI.Point | null = null;

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
        const centerX = this.embeddingPositions[i * 2];
        const centerY = this.embeddingPositions[i * 2 + 1];
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

      this.container.cursor = hoveredDatum ? 'pointer' : containerPointerDownPos ? 'grabbing' : 'default';
    });

    this.container
      .on('pointerdown', (evt: PIXI.InteractionEvent) => {
        // Ignore right clicks
        if ((evt.data.originalEvent as PointerEvent).button === 2) {
          return;
        }

        this.container.cursor = 'grabbing';
        containerPointerDownPos = evt.data.getLocalPosition(this.app.stage);

        if (hoveredDatum) {
          this.handlePointerDown(hoveredDatum);
        }
      })
      .on('pointerup', (evt: PIXI.InteractionEvent) => {
        this.container.cursor = 'default';
        const newPos = evt.data.getLocalPosition(this.app.stage);
        if (hoveredDatum || newPos.x !== containerPointerDownPos?.x || newPos.y !== containerPointerDownPos?.y) {
          containerPointerDownPos = null;
          return;
        }
        containerPointerDownPos = null;

        this.setSelectedAnimeID(null);
      });

    this.setColorBy(this.colorBy);

    this.renderNodes();

    this.updateLabels();

    this.renderLegend();
  }

  private getNodeRadiusAdjustment = (newZoomScale: number) => {
    let adjustment = (1 / (newZoomScale / 16)) * 1;
    adjustment = (adjustment + adjustment + adjustment + 1 + 1) / 5;
    adjustment = Math.min(adjustment, 1.5);
    return adjustment;
  };

  private handlePointerOver = (datum: EmbeddedPointWithIndex) => {
    this.maybeRemoveHoverObjects();
    const label = this.buildHoverLabel(datum);
    this.hoverLabelsContainer.addChild(label);

    const neighborLines = this.buildNeighborLines(datum);
    if (neighborLines) {
      this.decorationsContainer.addChild(neighborLines);
    }

    this.renderedHoverObjects = { label, neighborLines };
  };
  private handlePointerOut = () => this.maybeRemoveHoverObjects();
  private handlePointerDown = (datum: EmbeddedPoint) => this.setSelectedAnimeID(datum.metadata.id);

  private getNodeTexture = (): PIXI.Texture => {
    if (this.cachedNodeTexture) {
      return this.cachedNodeTexture;
    }

    const nodeGraphics = new this.PIXI.Graphics();
    nodeGraphics.lineStyle(10, 0xffffff, 1);
    nodeGraphics.beginFill(0xffffff);
    nodeGraphics.drawCircle(0, 0, BASE_RADIUS);
    nodeGraphics.endFill();
    const texture = this.app.renderer.generateTexture(nodeGraphics, {
      resolution: 5,
      scaleMode: this.PIXI.SCALE_MODES.LINEAR,
      multisample: this.PIXI.MSAA_QUALITY.MEDIUM,
    });
    this.cachedNodeTexture = texture;
    return texture;
  };

  private buildNodeSprite = (texture: PIXI.Texture, point: EmbeddedPoint) => {
    const radius = AtlasViz.getNodeRadius(point.metadata.rating_count);
    const nodeSprite = new this.PIXI.Sprite(texture);
    nodeSprite.anchor.set(0.5, 0.5);
    nodeSprite.interactive = false;
    const color = this.getNodeColor(point);
    nodeSprite.tint = color;
    nodeSprite.position.set(point.vector.x, point.vector.y);
    nodeSprite.scale.set((radius / BASE_RADIUS) * this.getNodeRadiusAdjustment(this.container.scale.x));
    return nodeSprite;
  };

  private renderNodes() {
    // Remove and destroy all children
    this.pointsContainer.removeChildren().forEach((c) => c.destroy({ texture: false, children: true }));

    const texture = this.getNodeTexture();

    this.embedding.forEach((point) => {
      const nodeSprite = this.buildNodeSprite(texture, point);
      this.pointsContainer.addChild(nodeSprite);
    });
  }

  private renderSelectedNodeObjects(selectedAnimeID: number | null) {
    if (this.selectedNode) {
      this.selectedNode.node.destroy({ texture: null });
      this.selectedNodeContainer.removeChild(this.selectedNode.node);
      this.decorationsContainer.removeChild(this.selectedNode.background);
      if (this.selectedNode.connections) {
        this.decorationsContainer.removeChild(this.selectedNode.connections);
      }
    }

    if (selectedAnimeID == null) {
      return;
    }

    const point = this.embeddedPointByID.get(selectedAnimeID)!;
    const texture = this.getNodeTexture();
    const nodeSprite = this.buildNodeSprite(texture, point);
    nodeSprite.tint = SELECTED_NODE_COLOR;
    this.selectedNodeContainer.addChild(nodeSprite);

    const nodeBackgroundTexture = this.getNodeBackgroundTexture();
    const backgroundSprite = this.buildNodeBackgroundSprite(nodeBackgroundTexture, point, SELECTED_NODE_COLOR);
    backgroundSprite.scale.x *= 1.5;
    backgroundSprite.scale.y *= 1.5;
    this.decorationsContainer.addChild(backgroundSprite);

    const connections = this.buildNeighborLines(point);
    if (connections) {
      this.decorationsContainer.addChild(connections);
    }

    return { node: nodeSprite, background: backgroundSprite, connections };
  }

  public setColorBy(colorBy: ColorBy) {
    this.colorBy = colorBy;
    this.colorScaler = AtlasViz.createColorScaler(colorBy);
  }

  public flyTo(id: number) {
    this.setSelectedAnimeID(id);
    const { x, y } = this.embedding.find((p) => p.metadata.id === id)!.vector;
    this.container.animate({
      time: 500,
      position: { x, y },
      scale: 26,
      ease: (curTime: number, minVal: number, maxVal: number, maxTime: number): number => {
        // cubic ease in out
        const t = curTime / maxTime;
        const p = t * t * t;
        return minVal + (maxVal - minVal) * p;
      },
      callbackOnComplete: () => this.updateLabels(true),
    });
  }

  private getTextScale = () => {
    const currentScale = this.container.scale.x;
    const textSize = (1 / currentScale) * 12.5 + 0.163;
    return textSize / BASE_LABEL_FONT_SIZE;
  };

  private setLabelScale = (
    g: PIXI.DisplayObject,
    datum: EmbeddedPointWithIndex,
    adjustment: number,
    multiplier = 1
  ) => {
    g.transform.scale.set(this.getTextScale() * multiplier);
    const radius = this.cachedNodeRadii[datum.index] * adjustment;
    g.position.set(datum.vector.x, datum.vector.y - radius);
    const circleRadius = AtlasViz.getNodeRadius(datum.metadata.rating_count);
    g.position.y -= 12 * (1 / this.container.scale.x) + circleRadius * 0.0023 * this.container.scale.x;
  };

  private buildHoverLabel = (datum: EmbeddedPointWithIndex): PIXI.Graphics => {
    const text = datum.metadata.title;
    const textWidth = this.textMeasurerCtx.measureText(text).width;

    const g = new this.PIXI.Graphics();
    g.beginFill(0x111111, 0.9);
    g.drawRoundedRect(0, 0, textWidth + 10, 50, 5);
    g.endFill();
    g.interactive = false;
    g.interactiveChildren = false;

    (g as any).datum = datum;
    this.setLabelScale(g, datum, this.getNodeRadiusAdjustment(this.container.scale.x));
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
    return g;
  };

  private maybeRemoveHoverObjects = () => {
    if (!this.renderedHoverObjects) {
      return;
    }

    const { label, neighborLines } = this.renderedHoverObjects;

    label.parent?.removeChild(label);
    label.destroy({ children: true });

    if (neighborLines) {
      neighborLines.parent?.removeChild(neighborLines);
      neighborLines.destroy({ children: true });
    }

    this.renderedHoverObjects = null;
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

  public setNeighbors(neighbors: number[][]) {
    this.neighbors = neighbors;
  }

  private computeVisibleNodeIndices = () => {
    const bounds = this.container.getVisibleBounds();
    let count = 0;

    const boundsMidpointX = bounds.x + bounds.width / 2;
    const boundsMidpointY = bounds.y + bounds.height / 2;

    for (let nodeIx = 0; nodeIx < this.embedding.length; nodeIx++) {
      const x = this.embeddingPositions[nodeIx * 2];
      const y = this.embeddingPositions[nodeIx * 2 + 1];

      const isInView = x > bounds.x && x < bounds.x + bounds.width && y > bounds.y && y < bounds.y + bounds.height;
      if (!isInView) {
        continue;
      }

      // Figure out which corner of the viewport the node is in
      // 0: top left, 1: top right, 2: bottom left, 3: bottom right
      this.visibleNodesScratch[count] = nodeIx;
      const corner = +(x > boundsMidpointX) + +(y > boundsMidpointY) * 2;
      this.visibleNodesQuadIndices[count] = corner;

      count += 1;
    }

    return count;
  };

  private computeLabelsToDisplay = (retainedLabels: NodeLabel[], ignoreUnrenderTimes = false): NodeLabel[] => {
    const now = Date.now();

    const computeLabelTransformedBounds = (label: NodeLabel): PIXI.Rectangle => {
      // Avoid rendering labels on top of each other
      const transformedWidth = label.textWidth * label.transform.scale.x;
      const transformedHeight = LABEL_HEIGHT * label.transform.scale.y;

      // The origin of the label as at its center, so adjust x and y to put it in the top left corner
      const bounds = new this.PIXI.Rectangle(
        label.x - transformedWidth / 2,
        label.y - transformedHeight / 2,
        transformedWidth,
        transformedHeight
      );
      // Grow bounds slightly to enforce a bit of space between the labels
      bounds.x -= bounds.width * 0.25;
      bounds.width *= 1.5;
      bounds.y -= bounds.height * 0.5;
      bounds.height *= 2;

      return bounds;
    };

    const visibleNodeCount = this.computeVisibleNodeIndices();
    const labelsToRender: { label: NodeLabel; transformedBounds: PIXI.Rectangle }[] = retainedLabels.map((label) => ({
      label,
      transformedBounds: computeLabelTransformedBounds(label),
    }));
    const labelSizeAdjustment = this.getNodeRadiusAdjustment(this.container.scale.x);

    const bounds = this.container.getVisibleBounds();
    const boundsMidpointX = bounds.x + bounds.width / 2;
    const boundsMidpointY = bounds.y + bounds.height / 2;

    let totalScore = labelsToRender.length;
    const maxScore = this.screenSpacePx / 110_000 + 4;
    const maxScorePerQuad = (maxScore / 4) * 1.15 + 1;
    const scoresByCorner = [0, 0, 0, 0];

    for (const label of labelsToRender) {
      const x = this.embeddingPositions[label.label.datum.index * 2];
      const y = this.embeddingPositions[label.label.datum.index * 2 + 1];
      const corner = +(x > boundsMidpointX) + +(y > boundsMidpointY) * 2;
      scoresByCorner[corner] += 1;
    }

    for (let i = 0; i < visibleNodeCount; i++) {
      const nodeIx = this.visibleNodesScratch[i];

      if (!ignoreUnrenderTimes && now - this.labelUnrenderTimes[nodeIx] < LABEL_PIN_TIME_MS) {
        continue;
      }

      const cornerIx = this.visibleNodesQuadIndices[i];

      if (scoresByCorner[cornerIx] > maxScorePerQuad) {
        continue;
      }

      const node = this.embedding[nodeIx];
      const label = this.buildLabel(node, labelSizeAdjustment);
      label.renderTimeMs = now;

      const bounds = computeLabelTransformedBounds(label);
      if (labelsToRender.some(({ transformedBounds }) => transformedBounds.intersects(bounds))) {
        continue;
      }

      labelsToRender.push({ label, transformedBounds: bounds });
      totalScore += 1;
      scoresByCorner[cornerIx] += 1;

      if (totalScore > maxScore) {
        break;
      }
    }

    return labelsToRender.map(({ label }) => label);
  };

  private buildLabel = (datum: EmbeddedPointWithIndex, labelSizeAdjustment: number) => {
    const text = datum.metadata.title;
    const cachedTextSprite = this.cachedLabels.get(text);
    if (cachedTextSprite) {
      this.setLabelScale(cachedTextSprite, datum, labelSizeAdjustment, LABEL_SCALE_MULTIPLIER);
      return cachedTextSprite;
    }

    const textWidth = this.textMeasurerCtx.measureText(text).width;

    const label = new this.NodeLabel(
      text,
      {
        fontFamily: 'PT Sans',
        fontSize: BASE_LABEL_FONT_SIZE,
        fill: 0xcccccc,
        align: 'center',
      },
      textWidth
    );
    label.position.set(5 + textWidth / 2, 25);
    label.interactive = false;

    label.datum = datum;
    label.textWidth = textWidth;
    this.setLabelScale(label, datum, labelSizeAdjustment, LABEL_SCALE_MULTIPLIER);
    this.cachedLabels.set(text, label);

    return label;
  };

  private updateLabels = (clearPins = false) => {
    const oldLabels = this.labelsContainer.removeChildren() as NodeLabel[];
    // Do not destroy the removed labels since we cache them

    const expiredLabelIndices = new Set<number>();

    const now = Date.now();
    const bounds = this.container.getVisibleBounds();
    const retainedLabels = clearPins
      ? []
      : oldLabels.filter((label) => {
          if (now - label.renderTimeMs > LABEL_PIN_TIME_MS) {
            const index = label.datum.index;
            expiredLabelIndices.add(index);
            return false;
          }

          // Remove if node for this label is no longer visible
          const x = this.embeddingPositions[label.datum.index * 2];
          const y = this.embeddingPositions[label.datum.index * 2 + 1];
          const isInView = x > bounds.x && x < bounds.x + bounds.width && y > bounds.y && y < bounds.y + bounds.height;
          if (!isInView) {
            const index = label.datum.index;
            // Set to 0 so that the label can be considered for re-display as soon as it comes back into view
            this.labelUnrenderTimes[index] = 0;
            return false;
          }

          return true;
        });

    const labelsToRender = this.computeLabelsToDisplay(retainedLabels, clearPins);

    for (const label of labelsToRender) {
      // If the unpinned label was re-rendered anyway, no need to expire it
      expiredLabelIndices.delete(label.datum.index);
      this.labelsContainer.addChild(label);
    }

    // For unpinned labels that didn't score high enough, we expire them for a while to avoid them
    // popping back in again very soon after
    for (const index of expiredLabelIndices) {
      this.labelUnrenderTimes[index] = now;
    }
  };

  public dispose() {
    this.app.ticker.stop();
  }
}
