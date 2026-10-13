package org.acme.store;

/** An order as returned by the search API. */
public record OrderEntity(Long orderKey, String customerId, String state) {}
