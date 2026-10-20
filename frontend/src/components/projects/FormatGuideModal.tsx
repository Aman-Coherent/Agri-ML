"use client";

import { FileText } from "lucide-react";

import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from "@/components/ui/dialog";

export function FormatGuideModal() {
  return (
    <Dialog>
      <DialogTrigger
        render={
          <Button variant="link" size="sm" className="h-auto p-0 text-xs">
            <FileText className="mr-1 size-3.5" /> View required CSV format
          </Button>
        }
      />
      <DialogContent className="sm:max-w-lg">
        <DialogHeader>
          <DialogTitle>CSV Format Guide</DialogTitle>
          <DialogDescription>
            Your file must be UTF-8 encoded with a header row. Column names don&apos;t need to
            match exactly — we auto-detect common variants for each field below.
          </DialogDescription>
        </DialogHeader>

        <div className="space-y-3 text-sm">
          <div className="rounded-md border p-3">
            <p className="font-medium">Lead ID / Native ID <span className="text-destructive">*required*</span></p>
            <p className="text-muted-foreground text-xs mt-0.5">
              Your own record id, included in every export so results match back to your data.
              Matches headers like <code>Lead ID</code>, <code>Native ID</code>, <code>Lead No</code>,{" "}
              <code>Record ID</code>, <code>CRM ID</code>, <code>Company ID</code> or <code>id</code>.
            </p>
          </div>
          <div className="rounded-md border p-3">
            <p className="font-medium">Company name <span className="text-destructive">*required*</span></p>
            <p className="text-muted-foreground text-xs mt-0.5">
              e.g. &quot;Rasi Seeds&quot;. Matches headers like <code>company_name</code>,{" "}
              <code>Company</code>, <code>Company Name</code>, <code>Business Name</code>,{" "}
              <code>Organization</code>, <code>Supplier</code>, <code>Vendor</code>, or{" "}
              <code>Name</code>.
            </p>
          </div>
          <div className="rounded-md border p-3">
            <p className="font-medium">Website URL <span className="text-muted-foreground font-normal">(strongly recommended)</span></p>
            <p className="text-muted-foreground text-xs mt-0.5">
              e.g. &quot;rasiseeds.com&quot;. Matches <code>url</code>, <code>website</code>,{" "}
              <code>domain</code>, <code>homepage</code>, <code>link</code>, etc. The homepage and its
              product pages are read to find real roles and products. Without a website, the web is
              searched by company name + location instead.
            </p>
          </div>
          <div className="rounded-md border p-3">
            <p className="font-medium">Location <span className="text-muted-foreground font-normal">(recommended)</span></p>
            <p className="text-muted-foreground text-xs mt-0.5">
              e.g. &quot;Zürich, Switzerland&quot;. Matches <code>location</code>,{" "}
              <code>address</code>, <code>city</code>, <code>state</code>, <code>country</code>,
              etc. — if your file splits this across multiple columns (City, State, Country) we
              combine them automatically into one location per row.
            </p>
          </div>
        </div>

        <div className="rounded-md bg-muted p-3">
          <p className="text-xs font-mono text-muted-foreground">
            Lead ID,Company Name,City,Country,Website
            <br />
            L-1001,Rasi Seeds,Attur,India,https://www.rasiseeds.com
            <br />
            L-1002,Sri Lakshmi Fertilizers,Guntur,India,
          </p>
        </div>

        <p className="text-xs text-muted-foreground">
          Maximum 200,000 rows per upload. After uploading, you&apos;ll see exactly which columns
          were detected before launching the job. Rows missing a company name are skipped and
          reported after upload.
        </p>
      </DialogContent>
    </Dialog>
  );
}
