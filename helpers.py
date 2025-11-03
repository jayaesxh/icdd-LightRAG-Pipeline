from rdflib import Literal, Namespace
from rdflib.namespace import XSD

CT = Namespace("https://standards.iso.org/iso/21597/-1/ed-1/en/Container#")

def iri_for_filename(g_index, relative_filename: str):
    """
    Find the document IRI in the index by ct:filename.
    Tries typed literal first, then untyped.
    """
    for s in g_index.subjects(CT.filename, Literal(relative_filename, datatype=XSD.string)):
        return s
    for s in g_index.subjects(CT.filename, Literal(relative_filename)):
        return s
    return None
