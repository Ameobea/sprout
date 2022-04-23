import type { D3ZoomEvent } from 'd3';
import * as d3 from 'd3';

import type { MALUserAnimeListItem } from '../malAPI';
import type { EmbeddedPoint, Embedding } from '../routes/index';
import ColorLegend from './ColorLegend';

const width = 1200;
const height = 1200;

enum ColorBy {
  AiredFromYear = 'aired_from_year',
  AverageRating = 'average_rating',
}

export class AtlasViz {
  private embedding: Embedding;
  private embeddingByID: Map<number, EmbeddedPoint>;
  private colorBy = ColorBy.AverageRating;
  private svg: d3.Selection<SVGSVGElement, unknown, HTMLElement, any>;
  private rootContainer: d3.Selection<d3.BaseType, unknown, HTMLElement, any>;
  private pointsContainer: d3.Selection<d3.BaseType, unknown, HTMLElement, any>;
  private decorationsContainer: d3.Selection<d3.BaseType, unknown, HTMLElement, any>;
  private labelsContainer: d3.Selection<d3.BaseType, unknown, HTMLElement, any>;
  private colorScaler: d3.ScaleSequential<string, never>;
  private renderedHoverLabel: number | null = null;
  private scaleX: d3.ScaleLinear<number, number>;
  private scaleY: d3.ScaleLinear<number, number>;
  private currentTransform: d3.ZoomTransform = d3.zoomIdentity
    .translate(window.innerWidth / 2, window.innerHeight / 2)
    .scale(0.02);
  private zoom: d3.ZoomBehavior<Element, unknown>;
  private hoverLabelCounter = 1;
  private setSelectedAnimeID: (id: number | null) => void;

  public displayMALUser(allMALData: MALUserAnimeListItem[]) {
    const malData = allMALData.filter((d) => d.list_status.status !== 'plan_to_watch');
    // Clear all previously marked points
    document
      .querySelectorAll('.mal-user-point')
      .forEach((circle) => circle.classList.remove('mal-user-point'));
    document.querySelectorAll('.mal-user-point-background').forEach((circle) => circle.remove());

    malData.forEach((datum) => {
      const animeID = datum.node.id;
      const circle = document.querySelector(`circle[data-anime-id="${animeID}"]`);
      if (!circle) {
        return;
      }

      circle.classList.add('mal-user-point');
    });
    const insertedPointBackgrounds = this.decorationsContainer
      .selectAll()
      .data(malData.map((d) => this.embeddingByID.get(+d.node.id)).filter((x) => !!x))
      .enter()
      .append('circle')
      .attr('class', 'mal-user-point-background')
      .attr('fill', 'url(#red-transparent-gradient)')
      .attr('cx', (d) => this.scaleX(d.vector[0]))
      .attr('cy', (d) => this.scaleY(d.vector[1]))
      .attr('r', (d) => {
        return 1600 + AtlasViz.getNodeRadius(d.metadata.rating_count) * 2.8;
        // return 1400;
      });

    // TODO: Compute connections
  }

  private getHoverLabelID() {
    return this.hoverLabelCounter++;
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
        return d3.scaleSequential(d3.interpolatePlasma).domain([1980, 2020]);
      case ColorBy.AverageRating: {
        return d3.scaleSequential(d3.interpolateViridis).domain([3, 7.5]);
      }
    }
  };

  private static getNodeRadius = (ratingCount: number) => Math.pow(ratingCount, 0.75) / 28.5 + 48;

  private getTextSize = () => {
    const baseFontSize = (1 / this.currentTransform.k) * 13 + 4;
    const fontSize =
      Math.round(baseFontSize - (baseFontSize % (this.currentTransform.k > 0.5 ? 1 : 5))) + 10;
    return `${fontSize}px`;
  };

  constructor(
    containerID: string,
    embedding: Embedding,
    setSelectedAnimeID: (id: number | null) => void,
    onPointerDown: (evt: MouseEvent) => void
  ) {
    this.embedding = embedding;
    this.embeddingByID = new Map(embedding.map((p) => [+p.metadata.id, p]));
    this.setSelectedAnimeID = setSelectedAnimeID;
    this.svg = d3.select(`#${containerID}`);
    this.rootContainer = d3.select('g#root-container');
    this.labelsContainer = this.rootContainer.select('g#labels-container');
    this.pointsContainer = this.rootContainer.select('g#points-container');
    this.decorationsContainer = this.rootContainer.select('g#decorations-container');

    this.scaleX = d3.scaleLinear().domain([-1, 1]).range([0, width]);
    this.scaleY = d3.scaleLinear().domain([-1, 1]).range([0, height]);

    this.setColorBy(this.colorBy);

    // Render the data in chunks to avoid locking up the browser
    const chunkSize = 1000;
    for (let i = 0; i < embedding.length; i += chunkSize) {
      const chunk = embedding.slice(i, i + chunkSize);
      this.pointsContainer
        .selectAll()
        .data(chunk)
        .enter()
        .append('circle')
        .attr('data-anime-id', (d) => d.metadata.id)
        .attr('cx', (d) => this.scaleX(d.vector[0]))
        .attr('cy', (d) => this.scaleY(d.vector[1]))
        .attr('r', (d) => AtlasViz.getNodeRadius(d.metadata.rating_count))
        .style('fill', (d) => this.colorScaler(d.metadata[this.colorBy]))
        .on('mouseenter', (_evt, d) => this.renderLabel(d))
        .on('mouseleave', () => this.maybeRemoveLabel())
        .on('pointerdown', (evt, d) => {
          this.setSelectedAnimeID(d.metadata.id);
          onPointerDown(evt);
        });
    }

    this.installZoomHandler();

    this.renderLegend();
  }

  private installZoomHandler = () => {
    const handleZoom = ({ transform }: D3ZoomEvent<any, any>) => {
      const transformString = `translate(${transform.x}px, ${transform.y}px) scale(${transform.k})`;
      [this.pointsContainer, this.labelsContainer, this.decorationsContainer].forEach(
        (container) => {
          // Setting the `transform` style attribute seems faster than setting the SVG `transform` attribute
          const node = container.node() as SVGGElement;
          // node.setAttribute(
          //   'style',
          //   `transform: translate(${transform.x}px, ${transform.y}px) scale(${transform.k})`
          // );
          node.style.transform = transformString;
        }
      );

      const oldTextSize = this.getTextSize();
      this.currentTransform = transform;
      const newTextSize = this.getTextSize();
      if (oldTextSize !== newTextSize) {
        this.labelsContainer.selectAll('text').style('font-size', newTextSize);
      }
    };

    // Need to do some dirty things to get D3 to stop butchering out performance
    const svgNode = this.svg.node() as SVGSVGElement;
    svgNode.createSVGPoint = null;
    svgNode.getBoundingClientRect = () => ({ top: 0, left: 0 });
    console.log(svgNode);

    this.zoom = d3.zoom().on('zoom', handleZoom);
    this.svg.call(this.zoom).call(this.zoom.transform, this.currentTransform);
  };

  public setColorBy(colorBy: ColorBy) {
    this.colorBy = colorBy;
    this.colorScaler = AtlasViz.createColorScaler(colorBy);
    this.renderLegend();
  }

  public flyTo(id: number) {
    const [x, y] = this.embedding.find((p) => p.metadata.id === id)!.vector;
    const newTransform = d3.zoomIdentity
      .translate(window.innerWidth / 2, window.innerHeight / 2)
      .scale(0.1)
      .translate(-this.scaleX(x), -this.scaleY(y));
    this.svg.transition().duration(400).call(this.zoom.transform, newTransform);
    this.setSelectedAnimeID(id);
  }

  private renderLabel = (d: EmbeddedPoint) => {
    this.maybeRemoveLabel();
    this.renderedHoverLabel = this.getHoverLabelID();

    this.labelsContainer
      .append('text')
      .attr('filter', 'url(#solid)')
      .attr('x', this.scaleX(d.vector[0]))
      .attr('y', this.scaleY(d.vector[1]) - AtlasViz.getNodeRadius(d.metadata.rating_count) - 10)
      .attr('id', `atlas-viz-hover-label-${this.renderedHoverLabel}`)
      .attr('class', 'hover-label')
      .attr('text-anchor', 'middle')
      .attr('text-baseline', 'bottom')
      .style('font-size', this.getTextSize())
      .text(d.metadata.title);
  };

  private maybeRemoveLabel = () => {
    if (this.renderedHoverLabel === null) {
      return;
    }

    const label = document.getElementById(`atlas-viz-hover-label-${this.renderedHoverLabel}`);
    if (label) {
      d3.select(label)
        .transition()
        .ease(d3.easeCubicIn)
        // Speed up the transition when zoomed in since there is less context to absorb
        .duration(1000 - Math.min(this.currentTransform.k * 6200, 800))
        .style('opacity', 0)
        .remove();
    }
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
}
