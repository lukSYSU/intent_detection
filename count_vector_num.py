from pymilvus import Collection, MilvusException, connections, db, utility

def get_vector_count(collection_name):
    """查询 Milvus 集合中的向量数量"""
    # 连接到 Milvus
    # connections.connect(
    #     alias="default",
    #     user = '',
    #     password = '',
    #     db_name = 'default',
    #     uri="http://localhost:19530"  
    # )
    
    # 获取集合对象
    collection = Collection(name=collection_name)
    
    # 加载集合（确保集合已加载）
    collection.load()
    
    # 查询向量数量
    num_vectors = collection.num_entities
    print(f"集合 '{collection_name}' 中的向量数量: {num_vectors}")
    return num_vectors




def show_milvus_database_info():
    # db_name = 'default'
    # connections.connect(alias='show_conn', host="127.0.0.1", port=19530)

    print(f"***********show milvus info**************")
    existing_databases = db.list_database()
    print(f"---exist databases: {existing_databases}")

    for db_name in existing_databases:
        # Use the database context
        db.using_database(db_name)

        collections = utility.list_collections()
        print(f"---DB:【{db_name}】 exist collections:{collections}")

        for coll_name in collections:
            # 获取集合对象
            collection = Collection(name=coll_name)

            collection.flush()
            
            # 加载集合（确保集合已加载）
            collection.load()
            
            # 查询向量数量
            num_vectors = collection.num_entities
            print(f"-----集合 '【{coll_name}】' 中的向量数量: {num_vectors}")


def drop_database(db_name):
    try:
        existing_databases = db.list_database()
        if db_name in existing_databases:
            print(f"Database '{db_name}' already exists.")

            # Use the database context
            db.using_database(db_name)

            # Drop all collections in the database
            collections = utility.list_collections()
            for collection_name in collections:
                collection = Collection(name=collection_name)
                collection.drop()
                print(f"Collection '{collection_name}' has been dropped.")

            db.drop_database(db_name)
            print(f"Database '{db_name}' has been deleted.")
        else:
            print(f"Database '{db_name}' does not exist.")

    except MilvusException as e:
        print(f"An error occurred: {e}")



def drop_table(db_name, tbl_name):
    try:
        existing_databases = db.list_database()
        if db_name in existing_databases:
            print(f"Database '{db_name}' already exists.")

            # Use the database context
            db.using_database(db_name)

            # Drop all collections in the database
            collections = utility.list_collections()
            if tbl_name in collections:
                collection = Collection(name=tbl_name)
                collection.drop()
                print(f"Collection '{tbl_name}' has been dropped.")
        else:
            print(f"Database '{db_name}' does not exist.")

    except MilvusException as e:
        print(f"An error occurred: {e}")

def create_database(db_name):
    try:
        existing_databases = db.list_database()
        if db_name in existing_databases:
            print(f"Database '{db_name}' already exists.")

        else:
            print(f"Database '{db_name}' does not exist.")
            db.create_database(db_name)
            print(f"Database '{db_name}' created successfully.")
    except MilvusException as e:
        print(f"An error occurred: {e}")
    



if __name__ == '__main__':
    db_name = 'default'

    connections.connect(
        alias="default",
        user = '',
        password = '',
        db_name = db_name,
        uri="http://localhost:19530"
    )

    show_milvus_database_info()
   
    

    # drop_table(db_name='default', tbl_name="LangChainCollection")
    # print(f"****After drop****")
    # show_milvus_database_info()


