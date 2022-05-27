export class AsyncSemaphore {
  private maxCount: number;
  private curCount = 0;
  private queue: (() => void)[] = [];

  constructor(maxCount: number) {
    this.maxCount = maxCount;
  }

  public acquire(): Promise<void> | void {
    if (this.curCount < this.maxCount) {
      this.curCount += 1;
      return;
    }

    return new Promise((resolve) => {
      this.queue.push(resolve);
    });
  }

  public release(): void {
    if (this.curCount > 0) {
      if (this.queue.length > 0) {
        this.queue.shift()!();
      } else {
        this.curCount -= 1;
      }
    } else {
      console.error('Tried to release from semaphore that has no active acquisitions');
    }
  }

  public async with<T>(fn: () => T | Promise<T>) {
    await this.acquire();
    try {
      return await fn();
    } finally {
      this.release();
    }
  }

  public wrap<F extends (...args: any[]) => any>(fn: F): F {
    return ((...args: Parameters<F>) => this.with(() => fn(...args))) as F;
  }
}
