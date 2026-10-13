package org.acme.store;

import java.sql.Connection;
import java.sql.SQLException;

public class OrderDao {
  private static final String BASE = "SELECT ORDER_KEY, STATE FROM ";

  /** Counts orders in a state. */
  int count(Connection c, String prefix, boolean onlyOpen) throws SQLException {
    final var table = prefix + TableNames.ORDERS;
    final StringBuilder sb = new StringBuilder("SELECT COUNT(*) FROM ").append(table);
    if (onlyOpen) {
      sb.append(" WHERE STATE = 'OPEN'");
    }
    c.prepareStatement(sb.toString());
    c.prepareStatement(BASE + table + " WHERE ORDER_KEY = ?");
    c.prepareStatement("""
        DELETE FROM ORDER_ITEM WHERE ORDER_KEY = ?
        """);
    return 0;
  }

  String notSql() {
    return "hello " + "world";
  }
}
