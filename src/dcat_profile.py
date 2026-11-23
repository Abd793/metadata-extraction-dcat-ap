from typing import Any

"""
DCAT-AP.de 3.0 profile definitions.

This module defines the metadata fields used by the
Schema-Guided extraction approach.

The existing Open-Schema implementation is not modified.
"""

# 1. Dataset-level properties

DATASET_FIELDS = {
    "Title": {
        "resource_class": "dcat:Dataset",
        "property_URI": "dcterms:title",
    },
    "Description": {
        "resource_class": "dcat:Dataset",
        "property_URI": "dcterms:description",
    },
    "Publisher": {
        "resource_class": "dcat:Dataset",
        "property_URI": "dcterms:publisher",
    },
    "Keyword": {
        "resource_class": "dcat:Dataset",
        "property_URI": "dcat:keyword",
    },
    "Landing Page": {
        "resource_class": "dcat:Dataset",
        "property_URI": "dcat:landingPage",
    },
    "Related Resource": {
        "resource_class": "dcat:Dataset",
        "property_URI": [
            "dcterms:references",
            "dcterms:isReferencedBy",
            "dcterms:relation",
        ],
    },
    "Documentation": {
        "resource_class": "dcat:Dataset",
        "property_URI": "foaf:page",
    },
}

# 2. Distribution-level properties

DISTRIBUTION_FIELDS = {
    "Access URL": {
        "resource_class": "dcat:Distribution",
        "property_URI": "dcat:accessURL",
    },
    "License": {
        "resource_class": "dcat:Distribution",
        "property_URI": "dcterms:license",
    },
}

# 3. Allowed values

CONFIDENCE_LEVELS = {
    "high",
    "medium",
    "low",
}

STATUSES = {
    "present",
    "missing",
    "unclear",
    "not_applicable",
}

EVIDENCE_LOCATION_TYPES = {
    "line",
    "section",
}


# 4. Create a standardized metadata item


def create_metadata_item(
    field: str,
    value: Any = None,
    evidence: str | None = None,
    evidence_location: dict | None = None,
    confidence: str = "low",
    reason: str = "",
    status: str = "missing",
    property_uri: str | None = None,
    resource_name: str | None = None,
) -> dict:
    """
    Create one standardized metadata item.

    Dataset fields use DATASET_FIELDS.
    Distribution fields use DISTRIBUTION_FIELDS.

    property_uri is only needed when a field allows
    multiple possible properties, such as Related Resource.
    """

    # 1. Find the field definition
    if field in DATASET_FIELDS:
        definition = DATASET_FIELDS[field]

    elif field in DISTRIBUTION_FIELDS:
        definition = DISTRIBUTION_FIELDS[field]

    else:
        raise ValueError(f"Unknown metadata field: {field}")

    # 2. Get resource class
    resource_class = definition["resource_class"]

    # 3. Determine the property URI
    allowed_uris = definition["property_URI"]

    if isinstance(allowed_uris, list):
        if property_uri not in allowed_uris:
            raise ValueError(
                f"{field} requires property_uri to be one of: "
                f"{allowed_uris}"
            )
        selected_uri = property_uri
    else:
        selected_uri = allowed_uris

        if property_uri is not None and property_uri != selected_uri:
            raise ValueError(
                f"Invalid property URI for {field}: {property_uri}"
            )

    # 4. Check confidence and status
    if confidence not in CONFIDENCE_LEVELS:
        raise ValueError(f"Invalid confidence: {confidence}")

    if status not in STATUSES:
        raise ValueError(f"Invalid status: {status}")

    # 5. Check value and evidence according to status
    if status == "present":
        if value is None or value == "":
            raise ValueError("A present item must have a value.")

        if not evidence:
            raise ValueError("A present item must have evidence.")

        if not evidence_location:
            raise ValueError(
                "A present item must have an evidence location."
            )

    elif status == "missing":
        value = None
        evidence = None
        evidence_location = None
        confidence = "low"

    elif status == "unclear":
        value = None

        if not evidence or not evidence_location:
            raise ValueError(
                "An unclear item must include evidence and its location."
            )

        confidence = "low"

    elif status == "not_applicable":
        value = None

        if not evidence or not evidence_location:
            raise ValueError(
                "A not_applicable item needs explicit supporting evidence."
            )

    # 6. Build the standardized item
    item = {
        "field": field,
        "resource_class": resource_class,
        "property_URI": selected_uri,
        "value": value,
        "evidence": evidence,
        "evidence_location": evidence_location,
        "confidence": confidence,
        "reason": reason,
        "status": status,
    }

    # 7. Distribution fields must identify their distribution
    if resource_class == "dcat:Distribution":
        if not resource_name:
            raise ValueError(
                "Distribution fields require a resource_name."
            )

        item["resource_name"] = resource_name

    elif resource_name is not None:
        raise ValueError(
            "resource_name should only be used for Distribution fields."
        )

    return item


