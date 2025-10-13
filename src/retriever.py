from neo4j import GraphDatabase, Driver

class Neo4jRetriever:
    """
    Retrieves a compact subgraph around an obpr_BuildingApplication and returns
    a relationship-preserving text block for the LLM.
    Pure Cypher (no APOC). Choose hop distance (1 or 2) at call time.
    """

    def __init__(self, uri: str, user: str, password: str, database: str = "neo4j"):
        self._driver: Driver = GraphDatabase.driver(uri, auth=(user, password))
        self._db = database

        self._q1 = """
        WITH $app_id AS target
        MATCH (app:obpr_BuildingApplication {applicationId: target})
        OPTIONAL MATCH p1 = (app)-[*1..1]-(n1)
        WITH app,
             collect(DISTINCT n1) AS n_list,
             [p IN collect(DISTINCT p1) WHERE p IS NOT NULL | p] AS paths
        WITH app, n_list,
             reduce(r = [], p IN paths | r + relationships(p)) AS rels
        RETURN {
          nodes: [node IN ([app] + n_list) |
                    { id: elementId(node), labels: labels(node), properties: properties(node) }],
          relationships: [rel IN rels |
                    { startNode: elementId(startNode(rel)),
                      endNode:   elementId(endNode(rel)),
                      type:      type(rel),
                      properties: properties(rel) }]
        } AS graph
        """

        self._q2 = """
        WITH $app_id AS target
        MATCH (app:obpr_BuildingApplication {applicationId: target})
        OPTIONAL MATCH p1 = (app)-[*1..2]-(n1)
        WITH app,
             collect(DISTINCT n1) AS n_list,
             [p IN collect(DISTINCT p1) WHERE p IS NOT NULL | p] AS paths
        WITH app, n_list,
             reduce(r = [], p IN paths | r + relationships(p)) AS rels
        RETURN {
          nodes: [node IN ([app] + n_list) |
                    { id: elementId(node), labels: labels(node), properties: properties(node) }],
          relationships: [rel IN rels |
                    { startNode: elementId(startNode(rel)),
                      endNode:   elementId(endNode(rel)),
                      type:      type(rel),
                      properties: properties(rel) }]
        } AS graph
        """

    def close(self):
        self._driver.close()

    def _fmt_node(self, node_map) -> str:
        labels = ":".join(node_map.get("labels", []))
        props = node_map.get("properties", {}) or {}
        show = []
        for k, v in props.items():
            if v is None: continue
            s = str(v)
            if len(s) > 120: s = s[:120] + "..."
            show.append(f"{k}: '{s}'")
        inside = ", ".join(show)
        return f"NODE: (:{labels} {{ {inside} }})"

    def _format_graph_as_text(self, graph: dict) -> str:
        if not graph: return "No data."
        nodes = {n["id"]: n for n in graph.get("nodes", []) if isinstance(n, dict) and "id" in n}
        lines = ["--- Knowledge Graph Context ---"]
        for n in nodes.values(): lines.append(self._fmt_node(n))
        for r in graph.get("relationships", []):
            if not isinstance(r, dict): continue
            s = nodes.get(r.get("startNode")); e = nodes.get(r.get("endNode"))
            if s and e:
                sl = ":".join(s.get("labels", [])); el = ":".join(e.get("labels", []))
                lines.append(f"RELATIONSHIP: (:{sl})-[{r.get('type')}]->(:{el})")
        lines.append("--- End of Context ---")
        return "\n".join(lines)

    def retrieve(self, application_id: str, hops: int = 1) -> str:
        if hops not in (1, 2): hops = 1
        query = self._q1 if hops == 1 else self._q2
        with self._driver.session(database=self._db) as s:
            rec = s.run(query, app_id=application_id).single()
            return self._format_graph_as_text((rec and rec.get("graph")) or {})
