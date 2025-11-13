from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Literal

AnswerType = Literal["string","bool","date","number","enum"]

@dataclass
class FieldSpec:
    id: str
    question: str
    answer_type: AnswerType
    required: bool = False
    enum_values: Optional[List[str]] = None
    keywords: Optional[List[str]] = None  # retrieval hints

@dataclass
class ExtractionSchema:
    application_fields: List[FieldSpec]
    building_fields: List[FieldSpec]

def default_schema() -> ExtractionSchema:
    return ExtractionSchema(
        application_fields=[
            FieldSpec("application_reference",
                      "What is the planning application reference number?",
                      "string", True, keywords=["reference","application ref","ref:"]),
            FieldSpec("application_type",
                      "What is the type of application (e.g., full planning, variation/removal of condition, listed building)?",
                      "string", True, keywords=["application type","type of application","proposal type"]),
            FieldSpec("is_retrospective",
                      "Is the application retrospective?", "bool", False, keywords=["retrospective","already carried out"]),
            FieldSpec("authority_name",
                      "Which local planning authority is this submitted to?",
                      "string", False, keywords=["authority","council"]),
            FieldSpec("applicant_name",
                      "What is the applicant or applicant company's name?",
                      "string", False, keywords=["applicant","agent"]),
            FieldSpec("site_address",
                      "What is the site address (street, city/town, postcode)?",
                      "string", True, keywords=["address","site address","location","postcode"]),
        ],
        building_fields=[
            FieldSpec("building_use_type",
                      "What is the primary building use?",
                      "string", False, keywords=["use","residential","commercial","class"]),
            FieldSpec("number_of_storeys",
                      "How many storeys does the building have (if stated)?",
                      "number", False, keywords=["storeys","floors"]),
        ]
    )
