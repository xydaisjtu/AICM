import pprint
import logging

class TianGong_HumanLabel_Parser:
    """
    A parser for the human_label.txt provided by TianGong-ST
    Return:
        - relevance_queries: information per query in human_label.txt
    """

    @staticmethod
    def parse(label_filename):
        label_reader = open(label_filename, "r")
        logger = logging.getLogger("GACM")
        relevance_queries = []
        query_count = dict()
        previous_id = -1

        cnt = 0
        for line in label_reader:
            entry_array = line.strip().split()
            id = int(entry_array[0])
            task = int(entry_array[1])
            query = int(entry_array[2])
            result = int(entry_array[3])
            relevance = int(entry_array[4])
            
            # count query-doc pairs
            if not query in query_count:
                query_count[query] = dict()
                query_count[query][result] = 1
            elif not result in query_count[query]:
                query_count[query][result] = 1
            else:
                query_count[query][result] += 1

            # The first line of a sample query
            if id != previous_id:
                info_per_query = dict()
                info_per_query['id'] = id
                info_per_query['sid'] = task
                info_per_query['qid'] = query
                info_per_query['uids'] = [result]
                info_per_query['relevances'] = [relevance]
                relevance_queries.append(info_per_query)
                cnt += 1
                previous_id = id
            
            # The rest lines of a query
            else:
                relevance_queries[-1]['uids'].append(result)
                relevance_queries[-1]['relevances'].append(relevance)
                cnt += 1
        
        tmp = 0
        for key in query_count:
            for x in query_count[key]:
                tmp += query_count[key][x]
        assert tmp == 20000
        assert cnt == 20000
        logger.info('num of queries in human_label.txt: {}'.format(len(relevance_queries)))
        return relevance_queries
        