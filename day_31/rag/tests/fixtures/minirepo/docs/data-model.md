# Data model

The ORDERS table stores customer orders. Each order has items in the ORDER_ITEM table.

## Search index

The acme-order index in Elasticsearch mirrors the orders table for search; mapping fields are strict.
