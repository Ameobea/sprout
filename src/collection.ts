export enum CollectionType {
  Anime = 'anime',
  Manga = 'manga',
}

export const getTableNames = (collectionType: CollectionType) => {
  switch (collectionType) {
    case CollectionType.Anime:
      return { listsTableName: '`mal-user-animelists`', collectedStatusColumnName: 'collected' };
    case CollectionType.Manga:
      return { listsTableName: '`mal-user-mangalists`', collectedStatusColumnName: '`collected-manga`' };
    default:
      throw new Error('Unknown collection type');
  }
};
