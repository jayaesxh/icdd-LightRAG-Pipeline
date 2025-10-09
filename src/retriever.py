# src/retriever.py
from neo4j import GraphDatabase, Driver

class Neo4jRetriever:
    """
    Retrieves a compact subgraph around an obpr_BuildingApplication and returns
    a relationship-preserving text block for the LLM.
    Pure Cypher (no APOC), with selectable hop distance (1 or 2).
    """

    def __init__(self, uri: str, user: str, password: str, database: str = "neo4j", hops: int = 1):
        self._driver: Driver = GraphDatabase.driver(uri, auth=(user, password))
        self._db = database
        self._hops = 1 if int(hops) != 2 else 2  # only 1 or 2 supported

    def close(self):
        self._driver.close()

    def _q_1hop(self) -> str:
        # Application + all neighbors (1 hop)
        return """
        WITH $app_id AS target
        MATCH (app:obpr_BuildingApplication {applicationId: target})
        OPTIONAL MATCH (app)-[r1]-(n1)
        WITH app, collect(DISTINCT n1) AS ns, collect(DISTINCT r1) AS rs
        RETURN {
          nodes: [{id:id(app), labels:labels(app), properties:properties(app)}] +
                 [n IN ns | {id:id(n), labels:labels(n), properties:properties(n)}],
          relationships: [r IN rs | {startNode:id(startNode(r)), endNode:id(endNode(r)), type:type(r)}]
        } AS graph
        """

    def _q_2hop(self) -> str:
        # Application + neighbors + neighbors-of-neighbors (2 hops)
        return """
        WITH $app_id AS target
        MATCH (app:obpr_BuildingApplication {applicationId: target})
        OPTIONAL MATCH (app)-[r1]-(n1)
        OPTIONAL MATCH (n1)-[r2]-(n2)
        WITH app, collect(DISTINCT n1)+collect(DISTINCT n2) AS ns,
             collect(DISTINCT r1)+collect(DISTINCT r2) AS rs
        RETURN {
          nodes: [{id:id(app), labels:labels(app), properties:properties(app)}] +
                 [n IN ns | {id:id(n), labels:labels(n), properties:properties(n)}],
          relationships: [r IN rs | {startNode:id(startNode(r)), endNode:id(endNode(r)), type:type(r)}]
        } AS graph
        """

    def _format_graph_as_text(self, graph: dict) -> str:
        if not graph:
            return "No data."
        nodes = {n["id"]: n for n in graph.get("nodes", [])}
        lines = ["--- Knowledge Graph Context ---"]
        for n in nodes.values():
            labels = ":".join(n.get("labels", []))
            props  = n.get("properties", {}) or {}
            short = ", ".join(f"{k}: '{str(v)[:120]}'" for k, v in props.items() if v is not None)
            lines.append(f"NODE: (:{labels} {{ {short} }})")
        for r in graph.get("relationships", []):
            sn = nodes.get(r.get("startNode")); en = nodes.get(r.get("endNode"))
            if sn and en:
                sl = ":".join(sn.get("labels", [])); el = ":".join(en.get("labels", []))
                lines.append(f"REL: (:{sl})-[{r.get('type')}]->(:{el})")
        lines.append("--- End of Context ---")
        return "\n".join(lines)

    def retrieve(self, application_id: str) -> str:
        with self._driver.session(database=self._db) as s:
            q = self._q_1hop() if self._hops == 1 else self._q_2hop()
            rec = s.run(q, app_id=application_id).single()
            graph = rec and rec.get("graph")
            return self._format_graph_as_text(graph or {})
