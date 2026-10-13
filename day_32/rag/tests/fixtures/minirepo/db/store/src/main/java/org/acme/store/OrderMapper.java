package org.acme.store;

import java.util.List;
import org.apache.ibatis.annotations.Param;

/** Mapper for orders. */
public interface OrderMapper {
  /** Searches orders by filter. */
  List<OrderEntity> search(OrderFilter filter);

  void insert(OrderDbModel order);

  void purge(@Param("tableName") String tableName);
}
