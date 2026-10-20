"use client";

import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { Loader2, Package } from "lucide-react";

import { Badge } from "@/components/ui/badge";
import {
  Sheet,
  SheetContent,
  SheetDescription,
  SheetHeader,
  SheetTitle,
} from "@/components/ui/sheet";
import { UrlSourceBadge } from "@/components/projects/UrlSourceBadge";
import { useCompanyTable } from "@/hooks/useCompanyTable";
import { api } from "@/lib/api";
import type { Company, CompanyProduct } from "@/lib/api";
import { cn } from "@/lib/utils";

const COLUMNS = [
  { key: "native_id", label: "Lead ID", width: "w-28" },
  { key: "company_name", label: "Company", width: "w-52" },
  { key: "location", label: "Location", width: "w-36" },
  { key: "status", label: "Status", width: "w-24" },
  { key: "url_source", label: "Source", width: "w-32" },
  { key: "roles", label: "Entities & roles", width: "flex-1" },
  { key: "products", label: "Products", width: "w-20" },
];

const PRODUCT_SOURCE_LABEL: Record<CompanyProduct["source"], string> = {
  website: "Website",
  web_search: "Web search",
  inferred: "Inferred",
};

const STATUS_DOT: Record<string, string> = {
  pending: "bg-muted-foreground/40",
  running: "bg-blue-500 animate-pulse",
  done: "bg-emerald-500",
  failed: "bg-red-500",
};

export function CompanyTable({
  projectId,
  token,
  statusFilter,
}: {
  projectId: string;
  token: string | null;
  statusFilter?: string;
}) {
  const { parentRef, rows, total, virtualizer, virtualItems, isLoading } = useCompanyTable(
    projectId,
    token,
    statusFilter
  );
  const [selected, setSelected] = useState<Company | null>(null);

  return (
    <div className="rounded-lg border">
      <div className="flex items-center justify-between border-b bg-muted/40 px-3 py-2">
        {COLUMNS.map((col) => (
          <div key={col.key} className={cn("text-xs font-medium text-muted-foreground", col.width)}>
            {col.label}
          </div>
        ))}
      </div>

      {isLoading ? (
        <div className="flex items-center justify-center py-16 text-muted-foreground">
          <Loader2 className="size-5 animate-spin" />
        </div>
      ) : rows.length === 0 ? (
        <div className="flex items-center justify-center py-16 text-sm text-muted-foreground">
          No companies to show.
        </div>
      ) : (
        <div ref={parentRef} className="h-[520px] overflow-auto">
          <div style={{ height: virtualizer.getTotalSize(), position: "relative" }}>
            {virtualItems.map((virtualRow) => {
              const company = rows[virtualRow.index];
              if (!company) {
                return (
                  <div
                    key={virtualRow.key}
                    style={{
                      position: "absolute",
                      top: 0,
                      left: 0,
                      width: "100%",
                      height: virtualRow.size,
                      transform: `translateY(${virtualRow.start}px)`,
                    }}
                    className="flex items-center justify-center text-xs text-muted-foreground"
                  >
                    Loading more...
                  </div>
                );
              }

              return (
                <div
                  key={virtualRow.key}
                  style={{
                    position: "absolute",
                    top: 0,
                    left: 0,
                    width: "100%",
                    height: virtualRow.size,
                    transform: `translateY(${virtualRow.start}px)`,
                  }}
                  className="flex cursor-pointer items-center border-b px-3 hover:bg-muted/40"
                  onClick={() => setSelected(company)}
                >
                  <div className="w-28 truncate pr-2 font-mono text-xs text-muted-foreground">
                    {company.native_id || "—"}
                  </div>
                  <div className="w-52 truncate pr-2 text-sm font-medium">{company.company_name}</div>
                  <div className="w-36 truncate pr-2 text-xs text-muted-foreground">
                    {company.location || "—"}
                  </div>
                  <div className="w-24 flex items-center gap-1.5">
                    <span className={cn("size-2 rounded-full", STATUS_DOT[company.status])} />
                    <span className="text-xs capitalize text-muted-foreground">{company.status}</span>
                  </div>
                  <div className="w-32">
                    <UrlSourceBadge source={company.url_read_source} />
                  </div>
                  <div className="flex min-w-0 flex-1 items-center gap-1.5 pr-2">
                    {company.is_agriculture === false && (
                      <Badge variant="outline" className="shrink-0 font-normal text-amber-500 border-amber-500/30">
                        Non-agri
                      </Badge>
                    )}
                    {company.has_new_roles && (
                      <Badge variant="outline" className="shrink-0 font-normal text-blue-500 border-blue-500/30">
                        NEW
                      </Badge>
                    )}
                    {company.display_label ? (
                      <span className="truncate text-xs" title={company.display_label}>
                        {company.display_label}
                      </span>
                    ) : (
                      <span className="text-xs text-muted-foreground">—</span>
                    )}
                  </div>
                  <div className="w-20 flex items-center gap-1 text-xs text-muted-foreground">
                    <Package className="size-3.5" />
                    {company.products_count || 0}
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      )}

      <div className="border-t px-3 py-2 text-xs text-muted-foreground">
        Showing {rows.length.toLocaleString()} of {total.toLocaleString()} companies
      </div>

      <CompanyDetailSheet company={selected} token={token} onClose={() => setSelected(null)} />
    </div>
  );
}

function CompanyDetailSheet({
  company,
  token,
  onClose,
}: {
  company: Company | null;
  token: string | null;
  onClose: () => void;
}) {
  const { data, isLoading } = useQuery({
    queryKey: ["company-products", company?.id],
    queryFn: () => api.getCompanyProducts(token as string, company!.id),
    enabled: !!company && !!token,
  });
  const groups = groupByRole(data?.products ?? []);

  return (
    <Sheet open={!!company} onOpenChange={(open) => !open && onClose()}>
      <SheetContent className="w-full sm:max-w-xl overflow-y-auto">
        <SheetHeader>
          <SheetTitle>{company?.company_name}</SheetTitle>
          <SheetDescription>
            {company?.native_id ? `Lead ID ${company.native_id}` : ""}
            {company?.location ? ` · ${company.location}` : ""}
            {company?.url ? ` · ${company.url}` : ""}
          </SheetDescription>
        </SheetHeader>

        <div className="space-y-4 px-4 pb-6">
          {company?.error_message && (
            <div className="rounded-md border border-red-500/30 bg-red-500/10 p-3 text-sm text-red-500">
              {company.error_message}
            </div>
          )}

          {company?.description && (
            <div>
              <p className="text-xs font-medium text-muted-foreground mb-1">What they do</p>
              <p className="text-sm">{company.description}</p>
            </div>
          )}

          <div className="flex flex-wrap items-center gap-2">
            <UrlSourceBadge source={company?.url_read_source ?? null} />
            {company?.is_agriculture === false && (
              <Badge variant="outline" className="font-normal text-amber-500 border-amber-500/30">
                Not agriculture
              </Badge>
            )}
            {company?.classification_confidence != null && (
              <span className="text-xs text-muted-foreground">
                {Math.round(company.classification_confidence * 100)}% confidence
              </span>
            )}
          </div>

          <div>
            <p className="text-xs font-medium text-muted-foreground mb-2">
              Products & services ({data?.products.length ?? 0}), by role
            </p>
            {isLoading ? (
              <Loader2 className="size-4 animate-spin text-muted-foreground" />
            ) : groups.length > 0 ? (
              <div className="space-y-3">
                {groups.map((g) => (
                  <div key={`${g.entity}-${g.role}`} className="rounded-md border">
                    <div className="flex flex-wrap items-center gap-1.5 border-b bg-muted/40 px-3 py-2">
                      <span className="text-sm font-medium">{g.role}</span>
                      {g.isNew && (
                        <Badge variant="outline" className="font-normal text-blue-500 border-blue-500/30">
                          NEW
                        </Badge>
                      )}
                      <span className="text-xs text-muted-foreground">
                        {g.entity}
                        {g.businessTypes ? ` · ${g.businessTypes}` : ""}
                      </span>
                    </div>
                    <div className="divide-y">
                      {g.items.map((p, i) => (
                        <div key={i} className="flex items-center justify-between gap-2 px-3 py-1.5">
                          <p className="text-sm truncate">{p.name}</p>
                          <div className="flex shrink-0 gap-1">
                            <Badge variant="secondary" className="text-xs font-normal capitalize">
                              {p.kind}
                            </Badge>
                            <Badge
                              variant="outline"
                              className={cn(
                                "text-xs font-normal",
                                p.source === "inferred"
                                  ? "text-muted-foreground"
                                  : "text-emerald-500 border-emerald-500/30"
                              )}
                            >
                              {PRODUCT_SOURCE_LABEL[p.source] ?? p.source}
                            </Badge>
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>
                ))}
              </div>
            ) : (
              <p className="text-sm text-muted-foreground">No products found for this company yet.</p>
            )}
          </div>
        </div>
      </SheetContent>
    </Sheet>
  );
}

function groupByRole(products: CompanyProduct[]) {
  const groups = new Map<
    string,
    { entity: string; role: string; businessTypes: string; isNew: boolean; items: CompanyProduct[] }
  >();
  for (const p of products) {
    const key = `${p.entity}||${p.category}`;
    const group = groups.get(key) ?? {
      entity: p.entity,
      role: p.category,
      businessTypes: p.business_types,
      isNew: p.is_new_role,
      items: [],
    };
    group.items.push(p);
    groups.set(key, group);
  }
  return Array.from(groups.values());
}
